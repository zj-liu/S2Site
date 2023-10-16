import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from collections import defaultdict
from tqdm import tqdm
import os

import torch
import torch.nn

from networks import s2site
from utilities import io_utils, evaluate

import utilities.configuration as cfg
import utilities.dataloader as dl
from networks.other_components import *


def PPBS_Plot(model_id, general_config, head_preds=False):
    device = general_config.device
    ckpt = torch.load(general_config.save_path % (model_id + '_PPBS'), map_location=device)
    config = ckpt['config_setting'].get_setting()
    config.device = device
    dataset_table = pd.read_csv(config.dataset_table) if config.dataset_table is not None else None

    # Load models
    model, wrapper, info = s2site.initial_S2Site(config=config)
    model.load_state_dict(ckpt['model_state'])
    model.reset_device(device)

    test_probability = []
    test_predictions = []
    test_labels = []
    test_weights = []
    
    out = [(174, 218), (300, 10), (882, 91), (510, 20)]
    
    if config.nfeatures_aa == 320:
        esm_version = '8M'
    elif config.nfeatures_aa == 480:
        esm_version = '35M'
    elif config.nfeatures_aa == 640:
        esm_version = '150M'
    elif config.nfeatures_aa == 1280:
        esm_version = '650M'
    else:
        esm_version = '3B'
        
    for i in range(5, 9):
        dataset = dl.read_dataset(i, config=config, ESMv=esm_version, dataset_table=dataset_table)
        if config.with_atom:
            probability, predicts = evaluate.test_predicts(inputs=dataset.inputs, model=model, wrapper=wrapper, return_all=False, inputs_id=i)
        else:
            probability, predicts = evaluate.test_predicts(inputs=dataset.inputs[:4], model=model, wrapper=wrapper, return_all=False, inputs_id=i)
        test_probability.append(probability)
        test_predictions.append(predicts)
        test_labels.append(np.array([np.argmax(label, axis=1)[:config.Lmax_aa] for label in dataset.outputs]))
        test_weights.append(dataset.weight)
        print(f'Test Dataset: Completed until id {i}.               ')
        
        if head_preds:
            sample, L = out[i-5]
            print(f'sample: {sample} and L: {L} to {L+10}')
            print(probability[sample][L:L+10])
            print(predicts[sample][L:L+10], test_labels[i-5][sample][L:L+10])
        
    test_probability.append(np.concatenate(test_probability))
    test_predictions.append(np.concatenate(test_predictions))
    test_labels.append(np.concatenate(test_labels))
    
    if test_weights[0] is not None:
        test_weights.append(np.concatenate(test_weights))

    print(f'Test Dataset: Completed until Test (all)!')
    print('=============Performance=============')
    names = config.dataset_titles[5:] + ['all']
    for i in range(len(test_predictions)):
        print(f'{names[i]} accuracy = {accuracy_score(np.concatenate(test_labels[i]), np.concatenate(test_predictions[i]))}')

    print('The model\'s best epoch is at: %i with %.6f accuracy' % (ckpt['best_epoch'], ckpt['val_perform'][-1]))
    
    if not os.path.isdir('./plots/'):
        os.mkdir('./plots/')
    if not os.path.isdir('./plots/PPBS/'):
        os.mkdir('./plots/PPBS/')
    print('=======================Plots=======================')
    evaluate.plot_performance_graphs(ckpt, model_id)
        
    fig,ax = evaluate.make_curves(
            test_labels,
            test_probability,
            test_weights,
            config.dataset_titles[5:] + ['Test (all)'],
            title = 'Protein-protein binding site prediction: %s' % (model_id + '_PPBS'),
            figsize=(10, 10),
            margin=0.05,grid=0.1,fs=16)
    fig.savefig('./plots/PPBS/PR_curve_PPBS_%s.png' % model_id, dpi=300)
    
    if not os.path.isdir(general_config.model_output_folder):
        os.mkdir(general_config.model_output_folder)
    io_utils.save_pickle({'probability': test_probability, 'dataset_names': config.dataset_titles[5:]+['Test (all)'], 
                          'labels': test_labels, 'weights': test_weights}, 
                         filename=general_config.model_output_folder+model_id+'.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PPI binding sites')
    parser.add_argument('--model_ids', dest='model_ids', type=str, nargs='+', help='Name of the model(s)')
    parser.add_argument('--gpu_id', dest='device', type=str, default='0', help='GPU id to program')
    args = parser.parse_args()
    
    general_config = cfg.Model_SetUp(device=args.device).get_setting()
    
    for m in args.model_ids:
        print(f'======================= {m} Starting =======================')
        PPBS_Plot(m, general_config=general_config)
        print(f'======================= {m} Completed =======================')
        
