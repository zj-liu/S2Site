import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm, tqdm_notebook

import torch
from torch.utils import data


def get_varies_L(input_mx, ngroups, with_atom=True):
    # use 8 * 108 * 1
    # batch_lens = [[],[],[],[],[],[],[],[]]
    # for i in range(no_of_types):
    #     for g in range(groups):
    #         if i in [0, 3, 4, 7]:
    #             res = (train_mx[i][g] == 0).nonzero()
    #             if len(res) > 0:
    #                 l = int(res[0,0].numpy())
    #             else:
    #                 l = train_mx[i][g].shape[0]
    #             batch_lens[i].append(l)
    #         else:
    #             batch_lens[i].append(batch_lens[i-1][g])
    # train_dataset = dl.Padded_Dataset(x=train_x, y=train_y, mx=batch_lens, lens=train_lens)
    
    no_of_types = 8 if with_atom else 4
    
    # use 1 * 108 * 8    
    batch_lens = []
    for g in range(ngroups):
        group_lens = []
        for i in range(no_of_types):
            if i in [0, 3, 4, 7]:
                res = (input_mx[i][g] == 0).nonzero()
                if len(res) > 0:
                    l = int(res[0,0].numpy())
                else:
                    l = input_mx[i][g].shape[0]
            group_lens.append(l)
        batch_lens.append(group_lens)
    return batch_lens


def read_labels(input_file, nmax=np.inf, label_type='int'):
    list_origins = []
    list_sequences = []
    list_labels = []
    list_resids = []
    name2idx = {}

    with open(input_file, 'r') as f:
        count = 0
        for line in f:
            if (line[0] == '>'):
                if count == nmax:
                    break
                if count > 0:
                    list_origins.append(origin)
                    list_sequences.append(sequence)
                    list_labels.append(np.array(labels))
                    list_resids.append(np.array(resids))

                origin = line[1:-1]
                sequence = ''
                labels = []
                resids = []
                count += 1
            else:
                line_splitted = line[:-1].split(' ')
                resids.append(line_splitted[:2])
                sequence += line_splitted[2]
                if label_type == 'int':
                    labels.append(int(line_splitted[-1]))
                else:
                    labels.append(float(line_splitted[-1]))

    list_origins.append(origin)
    list_sequences.append(sequence)
    list_labels.append(np.array(labels))
    list_resids.append(np.array(resids))

    list_origins = np.array(list_origins)
    list_sequences = np.array(list_sequences)
    list_labels = np.array(list_labels)
    list_resids = np.array(list_resids)
    
    count = 0
    for i in range(len(list_origins)):
        name2idx[list_origins[i]] = i

    return list_origins, list_sequences, list_resids, list_labels, name2idx


def read_data(filename, subset=None, exclude=None):
    env = pickle.load(open(filename, 'rb'))
    if (subset is not None) | (exclude is not None):
        if subset is not None:
            keys = subset
        else:
            keys = list(env.keys())
        if exclude is not None:
            for l, key in enumerate(keys):
                if key in exclude:
                    del keys[l]

        env_ = dict([(key, env[key]) for key in keys])
    else:
        env_ = env
    return env_


class DataSet_Constructor(data.Dataset):
    def __init__(self, label_addr=None, dataset_addr=None):
        self.label = label_addr
        self.dataset_name = dataset_addr
        self.weight = None

        # read required x and y
        if self.label:
            self.list_names, self.list_seqs, self.list_resids, self.list_labels, self.name2idx = read_labels(self.label)
        if self.dataset_name:
            dataset = read_data(self.dataset_name)
            self.inputs, self.outputs, self.fails = dataset['inputs'], dataset['outputs'], dataset['failed_samples']
            
            if self.label:
                self.used_samples = []
                self.used_name2idx = {}
                pos = 0
                for i in range(len(self.list_names)):
                    if i not in self.fails:
                        self.used_samples.append(self.list_names[i])
                        self.used_name2idx[self.list_names[i]] = pos
                        pos += 1

    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, index):
        out = []
        if self.label:
            out.append(self.list_names[index])
            out.append(self.list_labels[index])
        if self.dataset_name:
            if index in self.fails:
                out.append('Failed')
            else:
                out.append('Exists')
        return out
    
    def get_seq(self, chain_name):
        if self.label:
            return self.list_seqs[self.name2idx[chain_name]]
        else:
            return 'Sequence storage not exists'
        


def read_dataset(dataset_index, config, dataset_table=None, ESMv='3B', label_exist=True):
    if label_exist:
        label_addr = config.label_path % config.dataset_names[dataset_index]
    else:
        label_addr = None
    if config.use_esm:
        if dataset_index == 0 and config.mode == '_PPBS':
            dataset_addr = config.dataset_path % config.dataset_names[dataset_index]
        else:
            dataset_addr = config.dataset_wESM % (ESMv, config.dataset_names[dataset_index])
    else:
        dataset_addr = config.dataset_path % config.dataset_names[dataset_index]
    
    dataset = DataSet_Constructor(dataset_addr=dataset_addr, label_addr=label_addr)
    dataset.weight = None
    if dataset_table is not None:
        weights = np.array(dataset_table['Sample weight'][ dataset_table['Set'] == config.dataset_titles[dataset_index] ] )
        weights = np.array([weights[b] for b in range(len(weights)) if not b in dataset.fails])
        dataset.weight = weights
    return dataset


# Data Preparation before DataLoader
class Padded_Dataset(data.Dataset):
    def __init__(self, x, y, lens, mx):
        self.x = x
        self.y = y
        self.mx = mx
        self.lens = lens

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, index):
        xs = []
        mxs = []
        for i in range(len(self.x)):
            xs.append(self.x[i][index])
            mxs.append(self.mx[i][index])
        return xs, self.y[index], mxs, self.lens[index]
    

class Baseline_Dataset(data.Dataset):
    def __init__(self, x, y, sample_weight=None, max_batch_L=1024, min_batch_L=64, num_layer_blocks=2, is_tensor=True, model_type='unet', mode='train', platform='py') -> None:
        super(Baseline_Dataset, self).__init__()
        self.model_type = model_type
        self.max_batch_L = max_batch_L
        self.min_batch_L = min_batch_L

        if self.model_type == 'rcnn':
            self.factor = 2 ** (num_layer_blocks - 1) * 3
        else: # as default
            self.factor = 2 ** num_layer_blocks

        self.mode = mode
        self.platform = platform

        self.feats = x[1]
        if y is None:
            self.gt = np.zeros(self.feats.shape[0], dtype=np.int)
        else:
            self.gt = y

        # Get lengths of each sample
        self.L = np.array([len(input_) for input_ in self.feats ])

        if sample_weight is None:
            self.weights = None
        else:
            self.weights = sample_weight/( (sample_weight*self.L).mean()/ self.L.mean() )
        
        if mode == 'train':
            self.padding(is_tensor=is_tensor)
        else:
            self.no_batch(is_tensor=is_tensor)

    def padding(self, is_tensor=True):
        # Get batch info
        self.batch_L = []
        self.batch_idx = []
        self.idx2batch = {}

        accumlated_L, L_list = 0, []
        idx_list = []

        sorted_idx = np.argsort(self.L)
        
        if self.platform == 'py':
            progress = tqdm((sorted_idx), total=len(sorted_idx))
        else:
            progress = tqdm_notebook((sorted_idx), total=len(sorted_idx))
        progress.set_description(f'Forming Batches')

        for idx in progress:
            if accumlated_L + self.L[idx] > self.max_batch_L and len(L_list) > 0:
                self.batch_idx.append(idx_list)

                max_L = min(max(L_list), self.max_batch_L)
                max_L = max_L if max_L > self.min_batch_L else self.min_batch_L
                max_L = max_L if max_L % self.factor == 0 else max_L + (self.factor - (max_L % self.factor))

                self.batch_L.append( (L_list, max_L)  )
                idx_list = [idx]
                accumlated_L = self.L[idx]
                L_list = [self.L[idx]]
            else:
                idx_list.append(idx)
                accumlated_L += self.L[idx]
                L_list.append(self.L[idx])
            self.idx2batch[idx] = len(self.batch_idx)

        if accumlated_L > 0:
            self.batch_idx.append(idx_list)

            max_L = min(max(L_list), self.max_batch_L)
            max_L = max_L if max_L > self.min_batch_L else self.min_batch_L
            max_L = max_L if max_L % self.factor == 0 else max_L + (self.factor - (max_L % self.factor))

            self.batch_L.append( (L_list, max_L)  )

        # Padding and forming the batch
        self.batch_x = []
        self.batch_mask = []
        self.batch_y = []

        if self.platform == 'py':
            progress = tqdm((self.batch_idx), total=len(self.batch_idx))
        else:
            progress = tqdm_notebook((self.batch_idx), total=len(self.batch_idx))

        progress.set_description(f'Padding and Masking')
        for i, one_batch_idx in enumerate(progress):
        # for i, one_batch_idx in enumerate(self.batch_idx):
            one_batch = []
            one_batch_mask = []
            one_batch_y = []
            
            _, max_L = self.batch_L[i]

            for idx in one_batch_idx:
                sample = self.feats[idx]
                if sample.shape[0] < max_L:
                    padding = np.zeros( (max_L-sample.shape[0], 2560)) 

                    mask = np.concatenate( (np.ones((sample.shape[0], 1)), padding[:, :1]), axis=0).transpose()
                    one_batch_mask.append(mask)

                    sample = np.concatenate((sample, padding), axis=0).transpose()
                    one_batch.append(sample)
                else:
                    one_batch.append(sample[:max_L].transpose())
                    one_batch_mask.append(
                                            np.ones((1, max_L))
                                            )
                if self.weights is None:
                    gt = self.gt[idx][:max_L]
                else:
                    gt = self.gt[idx][:max_L] * self.weights[idx]
                    
                if is_tensor:
                    one_batch_y.append(  torch.tensor(gt, dtype=torch.float)   )
                else:
                    one_batch_y.append(gt)

            as_batch = np.stack(one_batch)
            as_batch_mask = np.stack(one_batch_mask)

            if is_tensor:
                self.batch_x.append( torch.tensor(as_batch, dtype=torch.float) )
                self.batch_mask.append( torch.tensor(as_batch_mask, dtype=torch.float) )        
                
            else:
                self.batch_x.append(as_batch)
                self.batch_mask.append(as_batch_mask)
                
            self.batch_y.append(one_batch_y)

    def no_batch(self, is_tensor=True):
        self.x = []
        self.y = []
        if self.platform == 'py':
            inputs = tqdm((self.gt), total=len(self.gt))
        else:
            inputs = tqdm_notebook((self.gt), total=len(self.gt))
        inputs.set_description(f'Processing Data')


        if is_tensor:
            for i, gt in enumerate(inputs):
                gt_len = len(gt) if len(gt) > self.min_batch_L else self.min_batch_L
                
                gt_len = gt_len if gt_len % self.factor == 0 else gt_len + (self.factor - (gt_len % self.factor))

                if gt_len == len(gt):
                    self.x.append(
                        torch.tensor(self.feats[i].transpose(), dtype=torch.float)
                    )
                else:
                    padded_feat = np.concatenate(
                        (self.feats[i], np.zeros((gt_len-len(gt), self.feats[i].shape[1]))), 
                        axis=0
                        )
                    self.x.append(
                        torch.tensor(padded_feat.transpose(), dtype=torch.float)
                    )
                    
                if self.weights is not None:
                    self.y.append(
                        torch.tensor(gt * self.weights[i], dtype=torch.float)
                    )
                else:
                    self.y.append(
                        torch.tensor(gt, dtype=torch.float)
                    )
        else:
            for i, gt in enumerate(inputs):
                gt_len = len(gt) if len(gt) > self.min_batch_L else self.min_batch_L
                
                gt_len = gt_len if gt_len % self.factor == 0 else gt_len + (self.factor - (gt_len % self.factor))

                if len(gt) == gt_len:
                    self.x.append(self.feats[i].transpose())
                else:
                    padded_feat = np.concatenate(
                        self.feats[i], np.zeros((gt_len-len(gt), self.feats[i].shape[1]))
                    )
                    self.x.append(padded_feat.transpose())
                if self.weight is not None:
                    self.y.append(gt * self.weights[i])
                else:
                    self.y.append(gt)

    def __len__(self):
        if self.mode == 'train':
            return len(self.batch_idx)
        else:
            return len(self.L)


    def __getitem__(self, index):
        '''
        Test mode: single sample on esm features, its corresponding ground truth, length
        Train/Va; mode: batch esm features, its corresponding batch mask, ground truth, batch length list, max(batch length) <= self.max_batch_L
        '''
        if self.mode == 'train':
            return self.batch_x[index], self.batch_mask[index], self.batch_y[index], self.batch_L[index][0], self.batch_L[index][1]
        else:
            return self.x[index], self.y[index], self.L[index]


    def get_batch_len(self, index):
        if self.mode == 'train':
            return len(self.batch_idx[index])
        else:
            return len(self.L)

    def get_batch_idx(self, index):
        return self.batch_idx[index]
 
    def get_mode(self):
        return self.mode

    def which_batch(self, index):
        if self.mode == 'train':
            return self.idx2batch[index]
        else:
            return index


if __name__ == '__main__':
    import configuration as c1
    configs = c1.Model_SetUp()
    config = configs.get_setting()
    testing = 1
    dataset_table = pd.read_csv(config.dataset_table)
    dataset = read_dataset(dataset_index=testing, config=config, dataset_table=dataset_table)

    print('1')
    print('2')
