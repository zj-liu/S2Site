import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import utilities.configuration as cfg
import utilities.dataloader as dl
from utilities import io_utils, evaluate
from networks import s2site
from networks.other_components import *


def train(model_id, setting, ESMv='3B', class_reweight=None, tensor_board=False):
    config = setting.get_setting()
    # initializing the model
    model, wrapper, info = s2site.initial_S2Site(config=config)
    model_full_name = model_id + config.mode
    print(f'Model full name: {model_full_name}. atom={config.with_atom}, reg={config.reg}, reweight={config.reweight}')
    model_path = config.save_path % model_full_name
    
    dataset_table = pd.read_csv(config.dataset_table) if config.reweight else None
    
    if tensor_board:
        if not os.path.isdir('./tb'):
            os.mkdir('./tb')
        writer = SummaryWriter(log_dir=f'./tb/{model_full_name}', comment='_ppbs', filename_suffix=model_full_name, flush_secs=1)

    data_x = []
    data_y = []
    data_weight = []

    start = time.time()
    print('Train and Val dataset loading...')
    # train[0] + validation[1..4]
    for i in tqdm(range(5)):
        dataset = dl.read_dataset(dataset_index=i, config=config, dataset_table=dataset_table, ESMv=ESMv)
        if config.with_atom:
            data_x.append(dataset.inputs)
        else:
            data_x.append(dataset.inputs[:4])
        data_weight.append(dataset.weight)
        data_y.append(dataset.outputs)
        '''
        wrapper.fit Outputs
        (x, y): grouped_inputs and group_outputs - similar to inputs and outputs but merged several aa and atoms into a sample.
        group info:
        [0] masks_inputs, [1] masks_outputs: masks for the padded data in bool form
        [2] groups: list of tuples containing information about aa and atoms in each merged sample.
        [3] group_lens: list of ints containing information about the max length that each merged sample reaches.
        '''
    
    if config.use_esm:
        print('Loading aa features generated from ESM2...')
        samples = len(data_x[0][1])
        for s in tqdm(range(samples)):
            path = config.dataset_train_path % (ESMv, config.dataset_names[0], s)
            data_x[0][1][s] = io_utils.load_pickle(path)['inputs']
    
    assert model.nam_aa.linear.weight.shape[-1] == data_x[0][1][0].shape[-1], 'The size of the embedding is not equivalent to the number of features defined in the model'

    val_npx = [np.concatenate([data_x[i][j] for i in [1,2,3,4] ] ) for j in range( len(data_x[0]) ) ]
    val_npy = np.concatenate([data_y[i] for i in [1,2,3,4]])
    val_npw = np.concatenate([data_weight[i] for i in [1,2,3,4]]) if config.reweight else None

    # Train
    train_x, train_y, train_grouped_info = wrapper.fit(data_x[0], data_y[0], sample_weight=data_weight[0])
    train_mx, train_lens = train_grouped_info[0], train_grouped_info[-1]
    # Val
    val_x, val_y, val_grouped_info = wrapper.fit(val_npx, val_npy, sample_weight=val_npw)
    val_mx, val_lens = val_grouped_info[0], val_grouped_info[-1]
    
    
    if config.mask:
        train_dataset = dl.Padded_Dataset(x=train_x, y=train_y, mx=train_mx, lens=train_lens)
        val_dataset = dl.Padded_Dataset(x=val_x, y=val_y, mx=val_mx, lens=val_lens)
    else:
        batch_lens = dl.get_varies_L(train_mx, len(train_lens), config.with_atom)
        train_dataset = dl.Padded_Dataset(x=train_x, y=train_y, mx=batch_lens, lens=train_lens, mask=False)
        
        batch_lens = dl.get_varies_L(val_mx, len(val_lens), config.with_atom)
        val_dataset = dl.Padded_Dataset(x=val_x, y=val_y, mx=batch_lens, lens=val_lens, mask=False)
        
    # DataLoader for Model Training
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    end = time.time()
    print('Time to load train and val dataset:', end-start)

    # train+val set-up
    device = config.device
    has_val = config.val
    val_over_ = 1
    epochs = config.epochs
    
    # reweight of the 2 classes
    if class_reweight is not None:
        weights = class_reweight
        class_weights = torch.FloatTensor(weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()
    # default eps for tf=1e-7 while pytorch=1e-8
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, verbose=1, mode='min', threshold=0.001, threshold_mode='abs', cooldown=2)    
    

    train_losses = [-1]
    train_xentropies = [-1]
    train_perform = [-1]
    train_time = [-1]

    val_losses = [-1]
    val_xentropies = [-1]
    val_perform = [-1]
    val_time = [-1]

    # early stop
    stopper = EarlyStopping(patience=5, verbose=True, delta=0.001)

    model.reset_device(device)
    model.train()
    total = len(train_dataloader)
    
    iter_count = 0
    start = time.time()
    for epoch in range(epochs):
        train_loss = []
        train_xentropy = []
        train_accuracy = []
        train_start = time.time()

        batches = tqdm((train_dataloader), total=total)
        for x, y, mx, lens in batches:
            if config.mask:
                pred = model(x, mx)
            else:
                assert x[0].shape[0] == 1
                pred = model(x)

            # classification loss and accuracy
            loss = 0.0
            accuracy = 0.0
            no_of_samples = 0
            tmp_pred = torch.max(model.output_act(pred), dim=-1)[1].cpu().detach().numpy()
            tmp_gt = torch.max(y, dim=-1)[1].numpy()
            
            for sample in range(pred.shape[0]):
                loss = loss + loss_fn(pred[sample, :lens[sample]], y[sample, :lens[sample]].to(device))
                accuracy += accuracy_score(tmp_gt[sample, :lens[sample]], tmp_pred[sample, :lens[sample]])
                no_of_samples += 1

            loss = loss / no_of_samples
            train_accuracy.append(accuracy / no_of_samples)
            train_xentropy.append(loss.item())

            # regularization loss, lambda already added to the computation function
            if config.reg:
                for _, param in model.named_parameters():
                    if isinstance(param, ConstraintParameter) and param.regularizer is not None:
                        loss = loss + param.compute_regularization()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            for _, param in model.named_parameters():
                if isinstance(param, ConstraintParameter) and param.constraint is not None:
                    param.data = param.apply_constraint().data

            if has_val:
                batches.set_description(f'Train Epoch: {epoch+1}/{epochs}. Train Processing') 
                batches.set_postfix_str(f'Train Batch Loss | Xentropy | Accuracy: {np.mean(train_loss):.6f} | {np.mean(train_xentropy):.6f} | {np.mean(train_accuracy):.6f}. Prev Val Loss | Xentropy | Accuracy: {val_losses[-1]:.6f} | {val_xentropies[-1]:.6f} | {val_perform[-1]:.6f}. Prev Train & Val Time: {train_time[-1]:.2f}s & {val_time[-1]:.2f}s')
            else:
                batches.set_description(f'Train Epoch: {epoch+1}/{epochs}. Train Processing') 
                batches.set_postfix_str(f'Train Batch Loss | Xentropy | Accuracy: {np.mean(train_loss):.6f} | {np.mean(train_xentropy):.6f} | {np.mean(train_accuracy):.6f}. Prev Train Loss | Xentropy | Accuracy | Time: {train_losses[-1]:.6f} | {train_xentropies[-1]:.6f} | {train_perform[-1]:.6f} | {train_time[-1]:.2f}s')
            
            iter_count += 1
            
            if tensor_board:
                writer.add_scalars("Train Losses", {'Cross Entropy': np.mean(train_xentropy), 'Loss': np.mean(train_loss)}, iter_count)
                writer.add_scalar("Train Accuracy", np.mean(train_accuracy), iter_count)
            
        train_end = time.time()
        train_time.append(train_end-train_start)
        train_losses.append(np.mean(train_loss))
        train_xentropies.append(np.mean(train_xentropy))
        train_perform.append(np.mean(train_accuracy))

        if has_val and epoch % val_over_ == 0:
            loss, xentropy, accuracy, timing = evaluate.val_test(model, val_dataloader, loss_fn=loss_fn, device=device, current_train_epoch=epoch+1, epochs=epochs)
            val_losses.append(loss)
            val_xentropies.append(xentropy)
            val_perform.append(accuracy)
            val_time.append(timing)
            
            
            # Visualize necessary statistics
            # atom part
            if tensor_board:
                for name, param in list(model.named_parameters())[:11]:
                    if param.grad is not None:
                        writer.add_histogram(name + '_grad', param.grad, epoch+1)
                    writer.add_histogram(name + '_data', param, epoch+1)
                
                # atom+aa last 10 learnable weights (ignore nam, nem)
                for name, param in list(model.named_parameters())[-10:]:
                    if param.grad is not None:
                        writer.add_histogram(name + '_grad', param.grad, epoch+1)
                    writer.add_histogram(name + '_data', param, epoch+1)
                    
                writer.add_scalars("Train-Val Cross Entropy", {"Train": train_xentropies[-1], "Val": xentropy}, epoch+1) 
                writer.add_scalars("Train-Val Loss", {"Train": train_losses[-1], "Val": loss}, epoch+1)
                writer.add_scalars("Train-Val Accuracy", {"Train": train_perform[-1], "Val": accuracy}, epoch+1)
                
            if stopper(model=model, val_loss=xentropy, loop=epoch+1, other_metric=[accuracy, loss]):
                torch.save({'model_state': model.state_dict(), 
                        'optimizer_state': optimizer.state_dict(), 
                        'train_losses': train_losses[1:],
                        'val_losses': val_losses[1:],
                        'train_xentropies': train_xentropies[1:],
                        'val_xentropies': val_xentropies[1:], 
                        'train_perform': train_perform[1:],
                        'val_perform': val_perform[1:],
                        'train_time': train_time[1:],
                        'val_time': val_time[1:],
                        'scheduler': scheduler.state_dict(),
                        'val_over': val_over_,
                        'best_epoch': stopper.loop,
                        'config_setting': setting}, 
                         model_path)

            if stopper.early_stop:
                if tensor_board:
                    writer.close()
                break

            scheduler.step(xentropy)
    end = time.time()

    if has_val:
        print('Train + Val takes: {:.2f}s to complete'.format(end-start))
    else:
        print('Train takes: {:.2f}s to complete'.format(end-start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Geometric Neural Network to Predict PPI binding sites')
    parser.add_argument('--version', dest='version', type=int, help='Model version for the trained model', required=True)
    parser.add_argument('--seed', dest='use_seed', type=int, default=31252, help='Use seed')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='Dropout in the Geometric Neural Network when using ESM features')
    parser.add_argument('--noESM', dest='use_ESM', action='store_false', default=True, help='Use ESM features in training the network')
    parser.add_argument('--ESMv', dest='ESMv', type=str, default='3B', choices=['8M', '35M', '150M', '650M', '3B'], help='Version of ESM used in the ESM setting')
    parser.add_argument('--no_Atom', dest='use_atom', action='store_false', default=True, help='Use atomic level features in training the network')
    parser.add_argument('--no_Reg', dest='use_reg', action='store_false', default=True, help='Use specified layer regularization in training the network')
    parser.add_argument('--no_Reweight', dest='use_reweight', action='store_false', default=True, help='Use computed sample weight in training the network')
    
    parser.add_argument('--gpu_id', dest='device', default='cpu', type=str, help='Give a gpu id to train the network if available')
    parser.add_argument('--tensor_board', dest='tensor_board', action='store_true', default=False, help='Use tensorboard to monitor the training process')
    
    args = parser.parse_args()
    io_utils.restricted_float(args.dropout)
    
    if args.use_seed > -1:
        cfg.set_seed(args.use_seed)

    model_name = 'PPI_' + args.ESMv
    dropout = args.dropout

    setting = cfg.Model_SetUp(name=model_name, device=args.device, reweight=args.use_reweight, reg=args.use_reg)
    setting.get_setting().dropout = dropout
    setting.use_atom(args.use_atom)
    
    model_id = f'{model_name}v{args.version}'
    print(f'============ {model_id} Training ============')
    train(model_id, setting, ESMv=args.ESMv, class_reweight=None, tensor_board=args.tensor_board)
    print(f'============ {model_id} Completed ============')
    
