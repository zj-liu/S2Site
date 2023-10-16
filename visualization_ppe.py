import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score, precision_recall_curve, precision_score, recall_score, matthews_corrcoef
from tqdm import tqdm
import os

import torch
import torch.nn as nn

from utilities import io_utils, evaluate, dataloader
import utilities.configuration as cfg
from networks import s2site

def binary_confusion_matrix(labels, predictions, is_missing):
    matrix = np.zeros((2,2))
    for i in range(len(labels)):
        matrix[labels[~is_missing][i]][predictions[~is_missing][i]] += 1
    tn = matrix[0][0]
    tp = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    return tp, tn, fp, fn

def main(model_id, general_config, recall_value=None, verbose=False, save_fig=None):
    path = general_config.save_path % (model_id + general_config.mode)
    device = general_config.device
    
    ckpt = torch.load(path, map_location=device)    
    config = ckpt['config_setting'].get_setting()
    config.device = device
    print('Model from:', path)
    print('Trained on Lmax_aa:', config.Lmax_aa)
    config.Lmax_aa = 1485
    print('Test on Lmax_aa:', config.Lmax_aa)
    
    if verbose:
        print('Train Info:')
        print('Train Loss:', ckpt['train_losses'])
        print('Train Cross Entropy:', ckpt['train_xentropies'])
        print('Train Accuracy:', ckpt['train_perform'])
        print('Loss:', ckpt['val_losses'])
        print('Cross Entropy:', ckpt['val_xentropies'])
        print('Accuracy:', ckpt['val_perform'])
    
    model, wrapper, info = s2site.initial_S2Site(config=config)
    model.load_state_dict(ckpt['model_state'])
    model.reset_device(device)
    
    ESMv = '3B'
    if '650M' in model_id:
        ESMv = '650M'
    elif '150M' in model_id:
        ESMv = '150M'
    elif '35M' in model_id:
        ESMv = '35M'
    elif '8M' in model_id:
        ESMv = '8M'

    dataset = dataloader.read_dataset(1, config=config, ESMv=ESMv, dataset_table=None)
    title_sample_pair = np.array([[title, sample] for title, sample in zip([config.dataset_titles[1]]*len(dataset.used_samples),  dataset.used_samples)])
    probabilities, predicts = evaluate.test_predicts(inputs=dataset.inputs, model=model, wrapper=wrapper, return_all=False, inputs_id='test')

    test_probability = np.concatenate(probabilities)
    test_labels = np.concatenate(np.array([np.argmax(label, axis=1) for label in dataset.outputs]))

    labels_flat = test_labels
    test_predictions = np.concatenate(predicts)
    predictions_flat = test_probability
    is_nan = np.isnan(predictions_flat) | np.isinf(labels_flat)
    is_missing = np.isnan(labels_flat) | (labels_flat<0)
    count_nan = is_nan.sum()
    if count_nan > 0:
        print('Found %s nan predictions in subset %s'%(count_nan, 'Test') )
        predictions_flat[is_nan] = np.nanmedian(predictions_flat)

    # calculate the metrics
    precision, recall, pr_threshold = precision_recall_curve(labels_flat[~is_missing], predictions_flat[~is_missing])
    fpr, tpr, roc_threshold = roc_curve(labels_flat[~is_missing], predictions_flat[~is_missing])
    rocauc = roc_auc_score(labels_flat[~is_missing], predictions_flat[~is_missing])
    auprc = auc(recall, precision)
    
    rec = recall_score(labels_flat[~is_missing], test_predictions[~is_missing])
    mcc = matthews_corrcoef(labels_flat[~is_missing], test_predictions[~is_missing])
    prec = precision_score(labels_flat[~is_missing], test_predictions[~is_missing])
    accuracy = accuracy_score(labels_flat[~is_missing], test_predictions[~is_missing])
    tp, tn, fp, fn = binary_confusion_matrix(labels_flat, test_predictions, is_missing)
    specificity = tn/(tn+fp)
    
    if recall_value is not None:
        length = len(tpr)
        sr, er = length, length
        for i in range(length):
            if round(tpr[i], 3) == recall_value:
                if sr == length:
                    sr = i
                er = i
                
        if sr == length:
            for i in range(length):
                if round(tpr[i]+.0005, 3) == recall_value:
                    if sr == length:
                        sr = i
                    er = i
                elif round(tpr[i]-.0005, 3) == recall_value:
                    if sr == length:
                        sr = i
                    er = i
                    
        set_predictions = np.array([1 if x > np.mean(roc_threshold[sr:er+1]) else 0 for x in test_probability])
        
        set_rec = recall_score(labels_flat[~is_missing], set_predictions[~is_missing])
        set_mcc = matthews_corrcoef(labels_flat[~is_missing], set_predictions[~is_missing])
        set_prec = precision_score(labels_flat[~is_missing], set_predictions[~is_missing])
        set_accuracy = accuracy_score(labels_flat[~is_missing], set_predictions[~is_missing])
        set_specificity = 1 - np.mean(fpr[sr:er+1])
        
    if not os.path.isdir('./plots/'):
        os.mkdir('./plots/')
    if not os.path.isdir('./plots/PPeBS/'):
        os.mkdir('./plots/PPeBS/')
        
    fig, ax = plt.subplots(figsize=(20, 10))
    margin=0.05
    grid=0.1
    fs=25
    plt.subplot(121)
    plt.plot(fpr, tpr, color='C0', linewidth=2.0, 
             label=f'Specificity={specificity:.3f}\nROCAUC={rocauc:.3f}\nMCC={mcc:.3f}\nAccuracy={accuracy:.3f}')
    if recall_value is not None:
        plt.scatter([np.mean(fpr[sr:er+1])], [np.mean(tpr[sr:er+1])], color='C2', linewidth=2.0, 
                    label=f'FPR={np.mean(tpr[sr:er+1]):.3f}\nTPR={np.mean(fpr[sr:er+1]):.3f}\nSpecificity={set_specificity:.3f}\nMCC={set_mcc:.3f}\nAccuracy={set_accuracy:.3f}\nthreshold={np.mean(roc_threshold[sr:er+1]):.3f}')
    plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.xlim([0 - margin, 1 + margin])
    plt.ylim([0 - margin, 1 + margin])
    plt.grid()

    plt.legend(fontsize=fs)
    plt.xlabel('False Positive Rate', fontsize=fs)
    plt.ylabel('True Positive Rate', fontsize=fs)
    plt.title('Protein-peptide interaction: %s' % (model_id + config.mode), fontsize=fs)
    plt.tight_layout()
    
    plt.subplot(122)
    plt.plot(recall, precision, color='C1', linewidth=2.0, label=f'Recall={rec:.3f}\nPrecision={prec:.3f}\nAUPRC={auprc:.3f}')
    if recall_value is not None:
        plt.scatter([set_rec], [set_prec], color='C3', linewidth=2.0, label=f'Recall={set_rec:.3f}\nPrecision={set_prec:.3f}')
    plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
    plt.xlim([0 - margin, 1 + margin])
    plt.ylim([0 - margin, 1 + margin])
    plt.grid()

    plt.legend(fontsize=fs)
    plt.xlabel('Recall', fontsize=fs)
    plt.ylabel('Precision', fontsize=fs)
    plt.title('Protein-peptide interaction: %s' % (model_id + config.mode), fontsize=fs)
    plt.tight_layout()
    
    if save_fig is not None:
        plt.savefig(save_fig%model_id, dpi=300)
    return {'labels': [np.array([np.argmax(label, axis=1) for label in dataset.outputs])], 'probability': [probabilities], 'weights': [None]}, title_sample_pair, [dataset]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visulization of protein-peptide binding site predictions')
    parser.add_argument('--model_ids', dest='model_ids', type=str, nargs='+', help='Name of the model(s)')
    parser.add_argument('--recall_value', dest='recall_value', type=float, default=None, help='')
    parser.add_argument('--gpu_id', dest='device', type=str, default='cpu', help='GPU id to program')
    
    args = parser.parse_args()

    for model in args.model_ids:
        if 'tl' in model:
            general_config = cfg.Model_SetUp(name='PPeI', tl=True, device=args.device).get_setting()
        else:
            general_config = cfg.Model_SetUp(name='PPeI', device=args.device).get_setting()
        main(model_id=model, general_config=general_config, recall_value=args.recall_value, verbose=False, save_fig='./plots/PPeBS/%s.png')

    
