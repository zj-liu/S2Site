import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score
from tqdm import tqdm
import os
import random

import torch
import torch.nn

from networks import s2site
from utilities import io_utils, evaluate

import utilities.configuration as cfg
import utilities.dataloader as dl
from networks.other_components import *

def main(model_id, general_config, fold, device_ind, head_preds=False):
    test_probability = []
    test_predictions = []
    test_labels = []
    test_weights = []

    perform = (0.,0.)
    losses = (0., 0.)
    xentropies = (0., 0.)

    for i in range(fold-1, 5):
        if general_config.mode == '_PPBS':
            path = general_config.save_path % (model_id + general_config.mode)
            ckpt = torch.load(path)
            if ckpt['config_setting'].get_setting().use_esm:
                config = cfg.Model_SetUp(name='BCE', device=device_ind).get_setting()
            else:
                config = cfg.Model_SetUp(name='BCE_noESM', device=device_ind).get_setting()
        else:
            path = general_config.save_path % (model_id + (general_config.mode % (i+1) ) )
            ckpt = torch.load(path)
            ckpt['config_setting'].reset_environment(device=device_ind)
            config = ckpt['config_setting'].get_setting()
            perform = np.add(perform, [ckpt['train_perform'][-1], ckpt['val_perform'][-1]])
            losses = np.add(losses, [ckpt['train_losses'][-1], ckpt['val_losses'][-1]])
            xentropies = np.add(xentropies, [ckpt['train_xentropies'][-1], ckpt['val_xentropies'][-1]])
        
        print('path:', path)
        print('model name:', config.model_name)
        dataset_table = pd.read_csv(config.dataset_table) if config.dataset_table is not None else None
        
        device = config.device
        model, wrapper, info = s2site.initial_S2Site(config=config)
        model.load_state_dict(ckpt['model_state'])
        model.reset_device(device)

        dataset = dl.read_dataset(i, config=config, dataset_table=dataset_table)
        probability, predicts = evaluate.test_predicts(inputs=dataset.inputs, model=model, wrapper=wrapper, return_all=False, inputs_id=i)
        test_probability.append(probability)
        test_predictions.append(predicts)
        test_labels.append(np.array([np.argmax(label, axis=1)[:config.Lmax_aa] for label in dataset.outputs]))
        test_weights.append(dataset.weight)
        print(f'Test Dataset: Completed until id {i+1}.               ')

        if head_preds:
            out = [(418, 48), (362, 88), (278, 87), (763, 134), (483, 49)]
            #sample = random.randint(0, len(probability))
            #L = random.randint(0, len(probability[sample])-11)
            sample, L = out[i]
            print(f'sample: {sample} and L: {L} to {L+10}')
            print(probability[sample][L:L+10])
            print(predicts[sample][L:L+10], test_labels[i][sample][L:L+10])


    test_probability.append(np.concatenate(test_probability))
    test_predictions.append(np.concatenate(test_predictions))
    test_labels.append(np.concatenate(test_labels))
    test_weights.append(np.concatenate(test_weights))
    print(f'Test Dataset: Completed until Test (all)!')

    print('=============Performance=============')
    names = config.dataset_titles + ['all']
    for i in range(len(test_predictions)):
        print(f'{names[i]} accuracy = {accuracy_score(np.concatenate(test_labels[i]), np.concatenate(test_predictions[i]))}')
    print('The model\'s best epoch is at: %i with %.6f accuracy' % (ckpt['best_epoch'], ckpt['val_perform'][-1]))


    all_probability = test_probability[-1]
    all_predict = test_predictions[-1]
    all_gt = test_labels[-1]
    precs = 0
    preds = []
    for idx, sample in enumerate(all_probability):
        ind = np.argsort(sample)[::-1][:len(sample)//10]
        ppv_predict = np.zeros(shape=all_predict[idx].shape)
        ppv_predict[ind] = 1

        precs += precision_score(all_gt[idx], ppv_predict)
        preds.append(ppv_predict)
    print('Length(probability, pred labels): (%i, %i) and average precision over samples: %.6f' % (len(all_probability), len(preds), precs / (idx+1)) )
    print('Precision over all samples: %.6f' % ( precision_score(np.concatenate(all_gt), np.concatenate(preds)) )  )

    if not os.path.isdir('./plots/'):
        os.mkdir('./plots/')
    if not os.path.isdir('./plots/BCE/'):
        os.mkdir('./plots/BCE/')
    if not os.path.isdir('./plots/BCE_all/'):
        os.mkdir('./plots/BCE_all/')

    if general_config.mode != '_PPBS':
        perform = np.divide(perform, 5)
        losses = np.divide(losses, 5)
        xentropies = np.divide(xentropies, 5)
        print(f'Train-Val Performance of {model_id}. Accuracy: {perform}; Cross Entropy: {xentropies}; Loss: {losses}')

        group = general_config.mode.split('_')[1]
    else:
        group = 'BCEnoTL'
    
    path = './plots/BCE/PR_curve_%s_%s.png' % (group, model_id)
        
    fig,ax = evaluate.make_curves(
            test_labels,
            test_probability,
            test_weights,
            config.dataset_titles + ['Fold (all)'],
            title = 'B-cell epitope prediction: %s' % (model_id + '_' + group),
            figsize=(10, 10),
            margin=0.05,grid=0.1,fs=25)
    fig.savefig(path, dpi=300)

    fig,ax = evaluate.make_curves(
            test_labels[-1:],
            test_probability[-1:],
            test_weights[-1:],
            ['Fold (all)'],
            title = 'B-cell epitope prediction: %s' % (model_id + '_' + group),
            figsize=(10, 10),
            margin=0.05,grid=0.1,fs=25)
    fig.savefig('./plots/BCE_all/PR_curve_BCE_%s.png' % model_id, dpi=300)

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visulization of B-cell Epitopes')
    parser.add_argument('--model_ids', dest='model_ids', type=str, nargs='+', help='Name of the model(s)')
    parser.add_argument('--fold', dest='fold', type=int, default=1, help='Visualize starts from which fold')
    parser.add_argument('--gpu_id', dest='device', type=str, default='0', help='GPU id to program')
    args = parser.parse_args()
    
    for model in args.model_ids:
        if 'tl' in model:
            general_config = cfg.Model_SetUp(name='BCE', tl=True, device=args.device).get_setting()
        elif 'PPI' in model:
            general_config = cfg.Model_SetUp(name='PPI', device=args.device).get_setting()
        else:
            general_config = cfg.Model_SetUp(name='BCE', device=args.device).get_setting()
        main(model_id=model, general_config=general_config, fold=args.fold, device_ind=args.device, head_preds=False)
    
