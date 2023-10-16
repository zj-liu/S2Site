import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

import random

import torch
import torch.nn
from torch.utils.data import DataLoader

from networks import rcnn, unet
from utilities import io_utils, evaluate

import utilities.configuration as cfg
import utilities.dataloader as dl
from networks.other_components import *


def Baseline_PPBS_Plot(model_id, test_datasets, general_config, head_preds=False):
    path = general_config.save_path % (model_id + '_PPBS')
    ckpt = torch.load(path, map_location=device)
    config = ckpt['config_setting'].get_setting()
    config.device = general_config.device

    # Load models
    if  'UNet' in config.model_name:
        model_type = 'unet'
        print('Model: UNet at dropout - ', config.dropout)
        model = unet.UNet(in_features=config.nfeatures_aa, conv_features=config.nembedding_aa, num_layers=config.num_layer_blocks)
    else: # RCNN
        print('Model: RCNN at dropout - ', config.dropout)
        model_type = 'rcnn'
        model = rcnn.RCNN(in_channels=config.nfeatures_aa, hidden_channels=config.nembedding_aa, 
                            num_rcnn_blocks=config.num_layer_blocks, gru_layer=config.num_gru_layers, 
                            gru_dropout=config.gru_dropout, with_pool=config.with_pool)

    model.load_state_dict(ckpt['model_state'])
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()

    test_probability = []
    test_predictions = []
    test_labels = []
    test_weights = []

    acc = 0.0
    losses = 0.0
    for i in range(5, 9):
        dataset = test_datasets[i-5]
        test_set = dl.Baseline_Dataset(dataset.inputs, dataset.outputs, sample_weight=dataset.weight, mode='test', max_batch_L=config.Bmax, min_batch_L=config.Bmin, num_layer_blocks=config.num_layer_blocks, model_type=model_type)
        test_dataloader = DataLoader(test_set, shuffle=False, num_workers=config.workers)
        
        loss, accuracy, timing, probability, predicts = evaluate.no_batch_predict(model, test_dataloader, loss_fn=loss_fn, device=device, current_train_epoch=i-4, epochs=4)
        
        acc += accuracy
        losses += loss

        test_probability.append(probability)
        test_predictions.append(predicts)
        test_labels.append(np.array([np.argmax(label, axis=1) for label in dataset.outputs]))
        test_weights.append(dataset.weight)
        
        if head_preds:
            show_out = [(174, 218), (300, 10), (882, 91), (510, 20)]
            # sample = random.randint(0, len(probability))
            # L = random.randint(0, len(probability[sample])-11)
            sample, L = show_out[i-5]
            print(f'sample: {sample} and L: {L} to {L+10}')
            print(probability[sample][L:L+10])
            print(predicts[sample][L:L+10], test_labels[i-5][sample][L:L+10])

    print(f'Test Dataset: Completed until Test (all)!')
    
    print('=============Performance=============')
    test_probability.append(np.concatenate(test_probability))
    test_predictions.append(np.concatenate(test_predictions))
    test_labels.append(np.concatenate(test_labels))
    test_weights.append(np.concatenate(test_weights))
    
    names = config.dataset_titles[5:] + ['Test all']
        
    for i in range(len(test_predictions)):
        print(f'{names[i]} accuracy = {accuracy_score(np.concatenate(test_labels[i]), np.concatenate(test_predictions[i]))}')
    print('From fn: Acc = %f, Loss = %f' % (acc/4, losses/4))

    print('The model\'s best epoch is at: %i with %.6f accuracy' % (ckpt['best_epoch'], ckpt['val_perform'][-1]))
    
    if not os.path.isdir('./plots/'):
        os.mkdir('./plots/')
    if not os.path.isdir('./plots/PPBS/'):
        os.mkdir('./plots/PPBS/')
        
    print('=======================Plots=======================')
    evaluate.plot_performance_graphs(ckpt, model_id)
    fig, ax = evaluate.make_curves(
            test_labels,
            test_probability,
            test_weights,
            names,
            title = 'Protein-protein binding site prediction: %s' % model_id,
            figsize=(10, 10),
            margin=0.05,
            grid=0.1,
            fs=16)
    fig.savefig('./plots/PPBS/PR_curve_PPBS_Baseline_%s.png' % model_id, dpi=300)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visulization of binding site prediction using Baseline Methods')
    parser.add_argument('--model_ids', dest='model_ids', type=str, nargs='+', help='Name of the model(s). eg., UNetv0_2, RCNNv0wo_2_2_0')
    parser.add_argument('--gpu_id', dest='device', type=str, default='0', help='GPU id to program')
    args = parser.parse_args()

    general_config = cfg.Model_SetUp(args.model_ids[0], device=args.device).get_setting()
    device = general_config.device
    dataset_table = pd.read_csv(general_config.dataset_table) if general_config.dataset_table is not None else None
    test_datasets = []
    for i in tqdm(range(5, 9)):
        test_datasets.append(dl.read_dataset(i, config=general_config, dataset_table=dataset_table))

    for m in args.model_ids:
        print(f'======================= {m} Starting =======================')
        Baseline_PPBS_Plot(model_id=m, test_datasets=test_datasets, general_config=general_config, head_preds=False)
        print(f'======================= {m} Completed =======================')
