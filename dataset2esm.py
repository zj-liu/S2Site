import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

import torch
from esm.pretrained import esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D

import utilities.dataloader as dl
from utilities import io_utils
import utilities.configuration as cfg

import argparse


def dataset2esm(config, ESMv, dataset_id_start=None, dataset_id_end=None, head_weights=False, contact_map=False):
    if ESMv == '3B':
        model, alphabet = esm2_t36_3B_UR50D()
        repr_layers = 36
    elif ESMv == '8M':
        model, alphabet = esm2_t6_8M_UR50D()
        repr_layers = 6
    elif ESMv == '35M':
        model, alphabet = esm2_t12_35M_UR50D()
        repr_layers = 12
    elif ESMv == '150M':
        model, alphabet = esm2_t30_150M_UR50D()
        repr_layers = 30
    elif ESMv == '650M':
        model, alphabet = esm2_t33_650M_UR50D()
        repr_layers = 33
    
    aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V',  'W', 'Y', '-']
    dataset_table = pd.read_csv(config.dataset_table) if config.dataset_table is not None else None
    
    device = config.device
    model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    
    dataset_id_start = 0 if dataset_id_start is None else dataset_id_start
    dataset_id_end = len(config.dataset_names) if dataset_id_end is None else dataset_id_end

    # train needs to save as per .data per residue
    for i in range(dataset_id_start, dataset_id_end):
        dataset_id = i
        dataset = dl.read_dataset(dataset_index=dataset_id, config=config, dataset_table=dataset_table, ESMv=ESMv)
        print('Read data from %s' % config.dataset_names[dataset_id])

        L = dataset.inputs[1].shape[0]
        data_seqs = []
        for l in range(L):
            attr = dataset.inputs[1][l]
            seq = []
            r, c = attr.shape
            for i in range(r):
                for j in range(c):
                    if attr[i][j] == 1:
                        seq.append(aa[j])
                        break
            data_seqs.append((l, ''.join(seq)))

        batches = len(data_seqs)
        tokens_embeddings = []
        with torch.no_grad():
            for s in tqdm(range(batches)):
                labels, strs, tokens = batch_converter(data_seqs[s:s+1])
                results = model(tokens.to(device), repr_layers=[repr_layers], need_head_weights=head_weights, return_contacts=contact_map)
                token_rep = results['representations'][repr_layers][0, 1:len(strs[0])+1].detach().cpu().numpy()
                if token_rep.shape[0] != len(strs[0]):
                    print('error at', s)
                    break
                tokens_embeddings.append(token_rep)
                
                if dataset_id == 0 and config.mode == '_PPBS':
                    path = config.dataset_train_path % (ESMv, config.dataset_names[0], s)
                    
                    if not os.path.isdir( os.path.dirname(os.path.dirname(path)) ):
                        os.mkdir( os.path.dirname(os.path.dirname(path)) )
                    if not os.path.isdir(os.path.dirname(path)):
                        os.mkdir(os.path.dirname(path))
                        
                    io_utils.save_pickle({'inputs': token_rep}, filename=path)
                else:
                    dataset.inputs[1][s] = token_rep
        
        if config.mode != '_PPBS' or dataset_id != 0:
            path = config.dataset_wESM % (ESMv, config.dataset_names[dataset_id])
            
            print('Save to path:', path)
            if not os.path.isdir(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
            io_utils.save_pickle({'inputs': dataset.inputs, 'outputs': dataset.outputs, 'failed_samples': dataset.fails}, filename=path)
        else:
            print('Last sample save to path:', path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Converting the one-hot residue vector to ESM embeddings')
    parser.add_argument('--ESMv', dest='ESMv', default='3B', type=str, nargs='+', choices=['3B', '650M', '150M', '35M', '8M'])
    parser.add_argument('--interaction', dest='interaction', default='PPI', type=str, choices=['PPI', 'PAI', 'PPeI'])
    parser.add_argument('--dataset_id_start', dest='dataset_id_start', default=None, type=int)
    parser.add_argument('--dataset_id_end', dest='dataset_id_end', default=None, type=int)
    parser.add_argument('--gpu_id', dest='device', default=0, type=int)
    
    args = parser.parse_args()
    args.interaction ='BCE' if args.interaction == 'PAI' else args.interaction
    setting = cfg.Model_SetUp(args.interaction, device=args.device)
    
    config = setting.get_setting()
    config.use_esm = False 
    torch.hub.set_dir(config.esm2_folder)
    
    for esm in args.ESMv:
        print(f'Converting aa in datasets to ESM({esm}) embeddings')
        dataset2esm(config=config, ESMv=esm, dataset_id_start=args.dataset_id_start, dataset_id_end=args.dataset_id_end)
        
