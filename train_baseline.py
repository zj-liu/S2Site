import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
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
from networks import rcnn, unet
from networks.other_components import *


def train(model_id, setting, model_type='unet', ESMv='3B', tensor_board=False):
    config = setting.get_setting()
    # initializing the model
    model_full_name = model_id + config.mode
    model_path = config.save_path % model_full_name

    # model
    if model_type == 'unet':
        model = unet.UNet(in_features=config.nfeatures_aa, conv_features=config.nembedding_aa, num_layers=config.num_layer_blocks, dropout_rate=config.dropout)
        print('Model: UNet with dropout:', model.dropout)
    else:
        model = rcnn.RCNN(in_channels=config.nfeatures_aa, hidden_channels=config.nembedding_aa, 
                            num_rcnn_blocks=config.num_layer_blocks, gru_layer=config.num_gru_layers, 
                            gru_dropout=config.gru_dropout, with_pool=config.with_pool, dropout_rate=config.dropout)
        print('Model: RCNN with dropout:', model.dropout)
        
    # import pdb; pdb.set_trace()
    dataset_table = pd.read_csv(config.dataset_table) if config.reweight else None
    
    if tensor_board:
        if not os.path.isdir('./tb'):
            os.mkdir('./tb')
        writer = SummaryWriter(log_dir=f'./tb/{model_full_name}', comment=config.mode, filename_suffix=model_full_name, flush_secs=1)

    data_x = []
    data_y = []
    data_weight = []
    
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

    val_npx = [np.concatenate([data_x[i][j] for i in [1,2,3,4] ] ) for j in range( len(data_x[0]) ) ]
    val_npy = np.concatenate([data_y[i] for i in [1,2,3,4]])
    val_npw = np.concatenate([data_weight[i] for i in [1,2,3,4]]) if config.reweight else None
    
    cool_down = 2
    patience = 2
    class_reweight = None
        
    if tensor_board:
        # Val data for aucpr
        labels = np.array([np.argmax(label, axis=1) for label in val_npy])
        labels_flat = np.concatenate(labels)
        writer.add_histogram('labels', labels_flat)
        
    # Train
    train_dataset = dl.Baseline_Dataset(data_x[0], data_y[0], sample_weight=data_weight[0], mode='train', max_batch_L=config.Bmax, min_batch_L=config.Bmin, num_layer_blocks=config.num_layer_blocks, model_type=model_type)
    # Val
    val_dataset = dl.Baseline_Dataset(val_npx, val_npy, sample_weight=val_npw, mode='val', max_batch_L=config.Bmax, min_batch_L=config.Bmin, num_layer_blocks=config.num_layer_blocks, model_type=model_type)
        
    # DataLoader for Model Training
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
    print('Datasets Loaded')

    # train+val set-up
    device = config.device
    has_val = config.val
    val_over_ = 1
    epochs = config.epochs
    
    loss_fn = nn.CrossEntropyLoss()
    
    # default eps for tf=1e-7 while pytorch=1e-8
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, eps=1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=1, mode='min', threshold=0.001, threshold_mode='abs', cooldown=cool_down)
    

    train_losses = [-1]
    train_perform = [-1]
    train_time = [-1]

    val_losses = [-1]
    val_perform = [-1]
    val_time = [-1]

    # early stop
    stopper = EarlyStopping(patience=5, verbose=True, delta=0.001)

    model.to(device)
    model.train()
    total = len(train_dataloader)
    
    iter_count = 0
    start = time.time()
    for epoch in range(epochs):
        train_loss = []
        train_accuracy = []
        train_start = time.time()

        batches = tqdm((train_dataloader), total=total)
        for x, mx, y, lens, maxL in batches:
            pred = model(torch.squeeze(x, 0).to(device))

            # classification loss and accuracy
            loss = 0.0
            accuracy = 0.0
            no_of_samples = pred.shape[0]
            tmp_pred = torch.max(model.output_act(pred), dim=-1)[1].cpu().detach().numpy()
            
            for sample in range(no_of_samples):
                loss = loss + loss_fn(pred[sample, :lens[sample]], y[sample][0].to(device))
                accuracy += accuracy_score(torch.max(y[sample][0], dim=-1)[1].numpy(), tmp_pred[sample, :lens[sample]])

            loss = loss / no_of_samples
            train_accuracy.append(accuracy / no_of_samples)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if has_val:
                batches.set_description(f'Train Epoch: {epoch+1}/{epochs}. Train Processing') 
                batches.set_postfix_str(f'Train Batch Loss | Accuracy: {np.mean(train_loss):.6f} | {np.mean(train_accuracy):.6f}. Prev Val Loss | Accuracy: {val_losses[-1]:.6f} | {val_perform[-1]:.6f}. Prev Train & Val Time: {train_time[-1]:.2f}s & {val_time[-1]:.2f}s')
            else:
                batches.set_description(f'Train Epoch: {epoch+1}/{epochs}. Train Processing') 
                batches.set_postfix_str(f'Train Batch Loss | Accuracy | Time: {np.mean(train_loss):.6f} | {train_perform[-1]:.6f} | {train_time[-1]:.2f}s')
            
            iter_count += 1
            if tensor_board:
                writer.add_scalars("Train Losses", {'Loss': np.mean(train_loss)}, iter_count)
                writer.add_scalar("Train Accuracy", np.mean(train_accuracy), iter_count)
            
        train_end = time.time()
        train_time.append(train_end-train_start)
        train_losses.append(np.mean(train_loss))
        train_perform.append(np.mean(train_accuracy))

        if has_val and epoch % val_over_ == 0:
            loss, accuracy, timing, probability, predictions = evaluate.no_batch_predict(model, val_dataloader, loss_fn=loss_fn, device=device, current_train_epoch=epoch+1, epochs=epochs)
            val_losses.append(loss)
            val_perform.append(accuracy)
            val_time.append(timing)
            
            # Visualize necessary statistics
            if tensor_board:
                for name, param in list(model.named_parameters()):
                    if param.grad is not None:
                        writer.add_histogram(name + '_grad', param.grad, epoch+1)
                    writer.add_histogram(name + '_data', param, epoch+1)
                writer.add_scalars("Train-Val Loss", {"Train": train_losses[-1], "Val": loss}, epoch+1)
                writer.add_scalars("Train-Val Accuracy", {"Train": train_perform[-1], "Val": accuracy}, epoch+1)
                
            if stopper(model=model, val_loss=loss, loop=epoch+1, other_metric=accuracy):
                torch.save({'model_state': model.state_dict(), 
                        'optimizer_state': optimizer.state_dict(), 
                        'train_losses': train_losses[1:],
                        'val_losses': val_losses[1:],
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

            scheduler.step(loss)
    end = time.time()

    if has_val:
        print('Train + Val takes: {:.2f}s to complete'.format(end-start))
    else:
        print('Train takes: {:.2f}s to complete'.format(end-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline Model Training')
    parser.add_argument('--version', dest='version', type=int, help='Model version for the trained model', required=True)
    parser.add_argument('--ESMv', dest='ESMv', type=str, default='3B', choices=['8M', '35M', '150M', '650M', '3B'], help='Version of ESM used in the ESM setting')
    parser.add_argument('--seed', dest='use_seed', type=int, default=31252, help='Use seed')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='Dropout in the Network when using ESM features')
    parser.add_argument('--num_layer_blocks', dest='num_layer_blocks', type=int, default=2, help='Num of blocks in the Network')
    parser.add_argument('--num_gru_layers', dest='num_gru_layers', type=int, default=2, help='Num of GRU layers in each block of the RCNN model')
    parser.add_argument('--gru_dropout', dest='gru_dropout', type=float, default=0, help='Dropout in the GRU layer')
    
    parser.add_argument('--model_type', dest='model_type', default='unet', choices=['unet', 'rcnn'])
    parser.add_argument('--with_pool', dest='with_pool', action='store_true', default=False, help='Add pooling layers in each network block')
    parser.add_argument('--gpu_id', dest='device', default='cpu', type=str, help='Give a gpu id to train the network if available')  
    parser.add_argument('--tensor_board', dest='tensor_board', action='store_true', default=False, help='Use tensorboard to monitor the training process')
    
    args = parser.parse_args()
    io_utils.restricted_float(args.dropout)
    io_utils.restricted_float(args.gru_dropout)
    if args.use_seed > -1:
        cfg.set_seed(args.use_seed)
    
    model_name = 'UNet' if args.model_type == 'unet' else 'RCNN'
    
    if args.model_type == 'unet':
        model_id = f'{model_name}v{args.version}_{args.num_layer_blocks}'
    else:
        if args.with_pool:
            # eg., {RCNN}v{version}_{2LayerBlock}_{2GRULayer}_{0GruDropout}
            model_id = f'{model_name}v{args.version}_{args.num_layer_blocks}_{args.num_gru_layers}_{args.gru_dropout}'
        else:
            model_id = f'{model_name}v{args.version}wo_{args.num_layer_blocks}_{args.num_gru_layers}_{args.gru_dropout}'
            
    setting = cfg.Model_SetUp(name=model_id, device=args.device)
    setting.get_setting().dropout = args.dropout
    config = setting.get_setting()
    print(f'============ {model_id} Training ============')
    print(f'Bmin, Bmax: ({config.Bmin}, {config.Bmax})')
    train(model_id, setting, model_type=args.model_type, ESMv=args.ESMv, tensor_board=args.tensor_board)
    print(f'============ {model_id} Completed ============')
    
