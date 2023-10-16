import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, roc_curve
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader

from utilities import dataloader as dl
from networks.other_components import ConstraintParameter


def val_test(model, dataloader, loss_fn, device='cpu', current_train_epoch=-1, epochs=-1):
    model.eval()
    losses = []
    xentropies = []
    accuracy = []
    total = len(dataloader)
    batches = tqdm((dataloader), total=total)
    s = time.time()
    
    for x, y, mx, lens in batches:
        for i in range(len(x)):
            x[i] = x[i].to(device)
            if model.mask:
                mx[i] = mx[i].to(device)

        with torch.no_grad():
            pred = model(x, mx)

            # classification loss
            loss = 0.0
            acc = 0.0
            no_of_samples = 0
            tmp_pred = torch.max(model.output_act(pred), dim=-1)[1].cpu().detach().numpy()
            tmp_gt = torch.max(y, dim=-1)[1].numpy()
            for sample in range(pred.shape[0]):
                loss += loss_fn(pred[sample, :lens[sample]], y[sample, :lens[sample]].to(device))
                acc += accuracy_score(tmp_gt[sample, :lens[sample]], tmp_pred[sample, :lens[sample]])
                no_of_samples += 1
            
            loss /= no_of_samples
            xentropies.append(loss.item())

            # regularization loss
            for _, param in model.named_parameters():
                if isinstance(param, ConstraintParameter) and param.regularizer is not None:
                    loss += param.compute_regularization()

            losses.append(loss.item())
            accuracy.append(acc / no_of_samples)
            batches.set_description(f'Train Epoch: {current_train_epoch}/{epochs}. Val Processing') 
            batches.set_postfix_str(f'Val Batch Loss | Xentropy | Accuracy: {np.mean(losses):.6f} | {np.mean(xentropies):.6f} | {np.mean(accuracy):.6f}')
    e = time.time()
    model.train()
    return np.mean(losses), np.mean(xentropies), np.mean(accuracy), e-s


def test_predicts(inputs, model, wrapper, return_all=True, Ls=None, inputs_id=None, input_group=None, return_group=False):
    if not model.with_atom:
        inputs = inputs[:4]
        
    if wrapper.multi_inputs:
        Ls = [len(input_) for input_ in inputs[0] ]
        ninputs = len(inputs)
    else:
        Ls = [len(input_) for input_ in inputs]
        ninputs = 1
        
    if wrapper.multi_outputs:
        noutputs = len(wrapper.Lmax_output) if (isinstance(wrapper.Lmax_output,list) | isinstance(wrapper.Lmax_output, tuple) ) else 10
        if wrapper.multi_inputs:
            if isinstance(wrapper.Lmax_output,list) | isinstance(wrapper.Lmax_output,tuple):
                output2inputs = [wrapper.Lmax.index(Lmax_output) for Lmax_output in wrapper.Lmax_output]
            else:
                output2inputs = [0 for _ in range(noutputs)]
            Loutputs = [[len(input_) for input_ in inputs[output2input]] for output2input in output2inputs]
        else:
            Loutputs = [Ls for _ in range(noutputs)]
    else:
        Loutputs = Ls
        noutputs = 1

    if wrapper.verbose:
        print('Generating groups...', end='\r')
        
        
    if input_group is None:
        groups, _ = wrapper.group_examples(Ls)
    else:
        groups = input_group
        
    if wrapper.multi_outputs:
        group_outputs = []
        for n in range(noutputs):
            Loutputs_ = Loutputs[n]
            Lmax_output = wrapper.Lmax_output if isinstance(wrapper.Lmax_output, int) else wrapper.Lmax_output[n]
            group_outputs_ = []
            for group in groups:
                start = 0
                group_ = []
                for index,_,_ in group:
                    group_.append( (index,min(start,Lmax_output), min(start+Loutputs_[index],Lmax_output) ) )
                    start += Loutputs_[index]
                group_outputs_.append(group_)
            group_outputs.append(group_outputs_)
    else:
        group_outputs = groups
    grouped_inputs, masks_inputs = wrapper.group_and_padd(inputs, groups)

    if wrapper.verbose:
        print('Grouped %s examples in %s groups. '%(len(Ls), len(groups)), end='\r')

    tensor_inputs = []
    tensor_inp_masks = []
    for i in range(ninputs):
        tensor_inputs.append(torch.from_numpy(grouped_inputs[i]))
        tensor_inp_masks.append(torch.from_numpy(masks_inputs[i])) 

    ngroups = len(groups)
    ninps = len(grouped_inputs)
    
    predictions = []
    preds = []
    
    layer_feats = []
    nbh_feats = []
    
    model.eval()
    for i in range(ngroups):
        xs = []
        mxs = []
        for j in range(ninps):
            xs.append(torch.unsqueeze(tensor_inputs[j][i], 0))
            mxs.append(torch.unsqueeze(tensor_inp_masks[j][i], 0))
        if model.return_mid:
            pred, esm_layer_feats, esm_nbh_feats = model(xs, mxs)
            pred = model.output_act(pred)
            layer_feats.append(esm_layer_feats.cpu().detach().numpy())
            nbh_feats.append(esm_nbh_feats.cpu().detach().numpy())
        else:
            pred = model.output_act(model(xs, mxs))
        
        preds.append(pred.cpu().detach().numpy())
        predictions.append(torch.max(pred, dim=-1)[1].cpu().detach().numpy())
        print(f'Test batch {i+1}/{ngroups}                      ', end='\r')
    
    preds = np.concatenate(preds)
    predictions = np.concatenate(predictions)

            
    if model.return_mid:
        layer_feats = np.concatenate(layer_feats)
        nbh_feats = np.concatenate(nbh_feats)
    
    model.train()
    if wrapper.verbose:
        print('Ungrouping and unpadding...                      ', end='\r')
        
    outputs = wrapper.ungroup_and_unpadd(preds, group_outputs)
    predictions = wrapper.ungroup_and_unpadd(predictions, group_outputs)
    if model.return_mid:
        layer_feats = wrapper.ungroup_and_unpadd(layer_feats, group_outputs)
        nbh_feats = wrapper.ungroup_and_unpadd(nbh_feats, group_outputs)

    if (not return_all) & wrapper.multi_outputs:
        out = np.array([output_[:,1] for output_ in outputs[0]])
    elif (not return_all) & ~wrapper.multi_outputs:
        out = np.array([output_[:,1] for output_ in outputs])
    elif return_all & wrapper.multi_outputs:
        out = [np.array(outputs_) for outputs_ in outputs]
    else:
        out = np.array([output_ for output_ in outputs])

    if wrapper.verbose:
        print(f'Prediction done for {inputs_id}!                              ')
    
    if model.return_mid and return_group:
        return out, predictions, layer_feats, nbh_feats, groups
    if model.return_mid:
        return out, predictions, layer_feats, nbh_feats
    if return_group:
        return out, predictions, groups
    
    return out, predictions

def make_curves(
        all_labels,
        all_predictions,
        all_weights,
        subset_names,
        title = '',
        figsize=(10, 10),
        margin=0.05,
        grid=0.1,
        fs=25,
        output_format='aucpr'):

    nSubsets = len(subset_names)
    subsetColors = ['C%s' % k for k in range(nSubsets)]

    all_PR_curves = []
    all_AUCPRs = []
    all_ROC_curves = []
    all_ROCAUCs = []

    for i in range(nSubsets):
        labels = all_labels[i]
        predictions = all_predictions[i]
        if all_weights[0] is not None:
            weights = all_weights[i]
            weights_repeated = np.array([np.ones(len(label)) * weight for label, weight in zip(labels, weights)], dtype=np.object)
        labels_flat = np.concatenate(labels)
        predictions_flat = np.concatenate(predictions)
        is_nan = np.isnan(predictions_flat) | np.isinf(labels_flat)
        is_missing = np.isnan(labels_flat) | (labels_flat<0)
        count_nan = is_nan.sum()
        if count_nan > 0:
            print('Found %s nan predictions in subset %s'%(count_nan,subset_names[i]) )
            predictions_flat[is_nan] = np.nanmedian(predictions_flat)
            
        if all_weights[0] is not None:
            precision, recall, _ = precision_recall_curve(labels_flat[~is_missing], predictions_flat[~is_missing], 
                                                        sample_weight = np.concatenate(weights_repeated)[~is_missing] )
            fpr, tpr, _ = roc_curve(labels_flat[~is_missing], predictions_flat[~is_missing], 
                                    sample_weight = np.concatenate(weights_repeated)[~is_missing] )
        else:
            precision, recall, _ = precision_recall_curve(labels_flat[~is_missing], predictions_flat[~is_missing])
            fpr, tpr, _ = roc_curve(labels_flat[~is_missing], predictions_flat[~is_missing])
            
        all_PR_curves.append((precision, recall) )
        all_AUCPRs.append( auc(recall, precision) )
        all_ROC_curves.append((tpr, fpr))
        all_ROCAUCs.append(auc(fpr, tpr))
        
    fig, ax = plt.subplots(figsize=figsize)
    if output_format == 'aucpr':
        for i in range(nSubsets):
            ax.plot(all_PR_curves[i][1], all_PR_curves[i][0], color=subsetColors[i], linewidth=2.0,
                    label='%s (AUCPR= %.3f)' % (subset_names[i], all_AUCPRs[i]))
        plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.xlim([0 - margin, 1 + margin])
        plt.ylim([0 - margin, 1 + margin])
        plt.grid()

        plt.legend(fontsize=fs)
        plt.xlabel('Recall', fontsize=fs)
        plt.ylabel('Precision', fontsize=fs)
    else:
        for i in range(nSubsets):
            ax.plot(all_ROC_curves[i][1], all_ROC_curves[i][0], color=subsetColors[i], linewidth=2.0,
                    label='%s (ROCAUC= %.3f)' % (subset_names[i], all_ROCAUCs[i]))
        plt.xticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.yticks(np.arange(0, 1.0 + grid, grid), fontsize=fs * 2/3)
        plt.xlim([0 - margin, 1 + margin])
        plt.ylim([0 - margin, 1 + margin])
        plt.grid()

        plt.legend(fontsize=fs)
        plt.xlabel('False Positive Rate', fontsize=fs)
        plt.ylabel('True Positive Rate', fontsize=fs)
    plt.title(title,fontsize=fs)
    plt.tight_layout()
    
    return fig, ax


def plot_performance_graphs(ckpt_info, model_id, mode='PPBS', single=False, idx=None, AUCPR=None):
    plt.figure(figsize=(10,5))
    
    x_axis = np.arange(len(ckpt_info['val_losses']))
    plt.xticks(np.arange(len(ckpt_info['val_losses']), step=1))
    if idx is not None:
        name = f'{model_id}_{mode}_{idx}'
    else:
        name = f'{model_id}_{mode}'
    
    if single:
        plt.plot(x_axis, ckpt_info['val_perform'], label='Accuracy=%.6f'%ckpt_info['val_perform'][-1])
        plt.plot(x_axis, ckpt_info['val_losses'], label='Loss=%.6f'%ckpt_info['val_losses'][-1])
        if 'val_xentropies' in ckpt_info.keys():
            plt.plot(x_axis, ckpt_info['val_xentropies'], label='Cross Entropy=%.6f'%ckpt_info['val_xentropies'][-1])

        if AUCPR is not None:
             plt.plot(x_axis, AUCPR[:-2], label='AUCPR=%.6f'%AUCPR[-3])
        
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.legend()
        plt.title('Val Graph: %s' % name )
    else:
        label1 = 'Train=%.6f'
        label2 = 'Val=%.6f'
        if 'val_xentropies' in ckpt_info.keys():
            plt.subplot(131)
        else:
            plt.subplot(121)

        plt.plot(x_axis, ckpt_info['train_perform'], label=label1 % ckpt_info['train_perform'][-1])
        plt.plot(x_axis, ckpt_info['val_perform'], label=label2 % ckpt_info['val_perform'][-1])
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.title('Accuracy')
        plt.legend()

        if 'val_xentropies' in ckpt_info.keys():
            plt.subplot(132)
        else:
            plt.subplot(122)
            
        plt.plot(x_axis, ckpt_info['train_losses'], label=label1 % ckpt_info['train_losses'][-1])
        plt.plot(x_axis, ckpt_info['val_losses'], label=label2 % ckpt_info['val_losses'][-1])
        plt.xlabel('Epochs')
        plt.ylabel('Performance')
        plt.title('Losses')
        plt.legend()

        if 'val_xentropies' in ckpt_info.keys():
            plt.subplot(133)
            plt.plot(x_axis, ckpt_info['train_xentropies'], label=label1%ckpt_info['train_xentropies'][-1])
            plt.plot(x_axis, ckpt_info['val_xentropies'], label=label2%ckpt_info['val_xentropies'][-1])
            plt.xlabel('Epochs')
            plt.ylabel('Performance')
            plt.title('Cross Entropy Loss')
            plt.legend()

        plt.suptitle('Performance_Graphs: %s' % name)
        plt.subplots_adjust(wspace=0.2)
    
    
    plt.tight_layout()
    plt.savefig('./plots/%s/Performance_Graphs_%s.png' % (mode, name), dpi=300)
    plt.show()


def no_batch_predict(model, dataloader, loss_fn=None, device='cpu', current_train_epoch=-1, epochs=-1, return_all=False, no_truth=False):
    """
    without initial batch of inputs 
    """
    model.eval()
    losses = []
    accuracy = []
    total = len(dataloader)
    batches = tqdm((dataloader), total=total)

    probability = []
    predictions = []

    s = time.time()
    
    for x, y, L in batches:
        assert x.shape[0] == 1
        with torch.no_grad():
            pred = model(x.to(device))
            prob = model.output_act(pred)
            if return_all:
                probability.append(prob[0, :L].cpu().detach().numpy())
            else:
                probability.append(prob[0, :L, 1].cpu().detach().numpy())

            predictions.append(torch.max(prob[0, :L], dim=-1)[1].cpu().detach().numpy())
            if no_truth:
                continue
            # classification loss
            loss = loss_fn(pred[0, :L, :], y[0].to(device))

            losses.append(loss.item())
            accuracy.append(accuracy_score(torch.max(y[0], dim=-1)[1].numpy(), predictions[-1]))
            batches.set_description(f'Train Epoch: {current_train_epoch}/{epochs}. Val Processing') 
            batches.set_postfix_str(f'Val Batch Loss | Accuracy: {np.mean(losses):.6f} | {np.mean(accuracy):.6f}')
    e = time.time()
    model.train()
    if no_truth:
        return np.array(probability), np.array(predictions)
    return np.mean(losses), np.mean(accuracy), e-s, np.array(probability), np.array(predictions)

