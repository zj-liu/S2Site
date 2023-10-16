import ml_collections
import torch
import os

import numpy as np
import random


class Model_SetUp():
    # Model initial setting
    def __init__(self, name='PPI', reweight=True, reg=True, mask=True, 
                    num_of_threads=8, set_cuda=True, device=None, worker=8, val=True, val_over_=1, tl=False, 
                    local_root=None, dataset_root=None, structures_folder=None, download_noFound=False):
        self.model_setting_dict = { 'with_atom': True,
                                    'Lmax_aa':1024, 
                                    'K_aa':16,
                                    'K_atom':16,
                                    'K_graph':32,
                                    'Dmax_aa':11.,
                                    'Dmax_atom':4.,
                                    'Dmax_graph':13.,
                                    'N_aa':32,
                                    'N_atom':32,
                                    'N_graph':32,
                                    'nfeatures_aa':2560, 
                                    'nfeatures_atom':12,
                                    'nembedding_atom':12,
                                    'nembedding_aa':32,
                                    'nembedding_graph':1,
                                    'dense_pooling':64, 
                                    'nattentionheads_pooling':64,
                                    'nfilters_atom':128,
                                    'nfilters_aa':128,
                                    'nfilters_graph':2,
                                    'nattentionheads_graph':1,
                                    'filter_MLP':[32],
                                    'covariance_type_atom':'full',
                                    'covariance_type_aa':'full',
                                    'covariance_type_graph':'full',
                                    'activation':'relu',
                                    'coordinates_atom':['euclidian'],
                                    'coordinates_aa':['euclidian'],
                                    'frame_aa':'triplet_sidechain',
                                    'coordinates_graph':['distance', 'ZdotZ', 'ZdotDelta', 'index_distance'],
                                    'index_distance_max_graph':8,
                                    'l1_aa':0.,
                                    'l12_aa':2e-3,
                                    'l12group_aa':0.,
                                    'l1_atom':0.,
                                    'l12_atom':2e-3,
                                    'l12group_atom':0.,
                                    'l1_pool':0.,
                                    'l12_pool':2e-3,
                                    'l12group_pool':0.,
                                    'dropout':0.,
                                    'optimizer':'adam',
                                    'fresh_initial_values':False,
                                    'save_initial_values':True,
                                    'output' : 'classification',
                                    'n_init':2,
                                    'epochs' : 100,
                                    'batch_size':1,
                                    'nrotations':1
                                }
        self.cfg = ml_collections.ConfigDict(self.model_setting_dict)

        self.cfg.reweight = reweight
        self.cfg.reg = reg
        self.cfg.mask = mask
        self.cfg.tl = tl
        self.cfg.download_noFound = download_noFound
        self.cfg.use_esm = True

        # Train Setting
        self.reset_model(new_model=name, tl=tl, local_root=local_root, dataset_root=dataset_root, structures_folder=structures_folder)
        self.reset_environment(num_of_threads=num_of_threads, set_cuda=set_cuda, device=device, worker=worker, val=val, val_over_=val_over_)

 
    def reset_model(self, new_model=None, tl=False, 
                    local_root=None, dataset_root=None, structures_folder=None):
        self.model_list = set(['PPI', 'BCE', 'PPeI'])
        if new_model is not None:
            if new_model in self.model_list or new_model.split('_')[1] in ['8M', '35M', '150M', '650M', '3B']:
                self.cfg.model_name = new_model
            elif 'UNet' in new_model or 'RCNN' in new_model:
                self.cfg.model_name = new_model
            else:
                print(f'{new_model} is not a recorded model name. Configuration changed to \'Net\'. Please give a valid model name from {self.model_list}.')
                self.cfg.model_name = 'PPI'
                
            # for PPI default setting with atom
            self.cfg.with_atom = True
            self.cfg.Lmax_aa = 1024         # use 2120 if use bce else 1024
            self.cfg.nfeatures_aa = 2560    # use 2560 for esm-2(3B)
            self.cfg.dense_pooling = 64     # use 64 if with_atom else 0
        
        if '8M' in self.cfg.model_name:
            self.cfg.nfeatures_aa = 320
        elif '35M' in self.cfg.model_name:
            self.cfg.nfeatures_aa = 480
        elif '150M' in self.cfg.model_name:
            self.cfg.nfeatures_aa = 640
        elif '650M' in self.cfg.model_name:
            self.cfg.nfeatures_aa = 1280
        
        if tl:
            if 'BCE' in self.cfg.model_name:
                self.cfg.mode = '_PPBSTLBCE_%i'
                self.cfg.Lmax_aa = 2120
            elif 'PPe' in self.cfg.model_name:
                self.cfg.mode = '_PPBSTLPPeBS'
                self.cfg.Lmax_aa = 1485
            else:
                print('Did not match any default interactions with transfer learning mode.\nReturn to train from scratch with PPI')
                self.cfg.mode = '_PPBS'
        else:
            if 'BCE' in self.cfg.model_name:
                self.cfg.mode = '_BCE_%i'
                self.cfg.Lmax_aa = 2120
            elif 'PPe' in self.cfg.model_name:
                self.cfg.mode = '_PPeBS'
                self.cfg.Lmax_aa = 1485
            else:
                self.cfg.mode = '_PPBS'
        self.cfg.tl = tl

        if 'UNet' in self.cfg.model_name:
            self.cfg.num_layer_blocks = int(self.cfg.model_name.split('_')[1])
            factor = 2 ** self.cfg.num_layer_blocks
            self.cfg.Bmax = 1024 if (1024 % factor) == 0 else 1024 + (factor - (1024 % factor))
            self.cfg.Bmin = 8 * factor
        elif 'RCNN' in self.cfg.model_name:
            self.cfg.num_layer_blocks = int(self.cfg.model_name.split('_')[1])
            self.cfg.with_pool = False if self.cfg.model_name.split('_')[0][-2:] == 'wo' else True
            if self.cfg.with_pool:
                factor = 2 ** (self.cfg.num_layer_blocks-1) * 3 
            else:
                factor = 2 ** self.cfg.num_layer_blocks
                
            self.cfg.Bmax = 1024 if (1024 % factor) == 0 else 1024 + (factor - (1024 % factor))
            self.cfg.Bmin = 8 * factor
            self.cfg.num_gru_layers = int(self.cfg.model_name.split('_')[2])
            self.cfg.gru_dropout = float(self.cfg.model_name.split('_')[3])
            
        self.reset_path(local_root=local_root, dataset_root=dataset_root, structures_folder=structures_folder)
    # Folder Path Setting
    # Github and Data Path
    def reset_path(self, local_root=None, dataset_root=None, structures_folder=None):
        self.cfg.local_root = '' if local_root is None else local_root
        self.cfg.dataset_root = '/cto_studio/xtalpi_lab/zhanglinwei/' if dataset_root is None else dataset_root
        self.cfg.structures_folder = self.cfg.dataset_root + 'datasets/PDB/' if structures_folder is None else structures_folder
        # self.cfg.structures_folder = '/cto_studio/xtalpi_lab/liuzijing/ScanNet/PDB/'
        self.cfg.model_folder = self.cfg.local_root + 'models/'
        self.cfg.predicts = self.cfg.local_root + 'predictions/'
        
        self.cfg.pipeline_folder = self.cfg.dataset_root + 'datasets/pipelines/'
        self.cfg.model_output_folder = self.cfg.dataset_root + 'datasets/model_outputs/'
        self.cfg.esm2_folder = self.cfg.dataset_root + 'pretrained_model/hub'
        
        if 'UNet' in self.cfg.model_name or 'RCNN' in self.cfg.model_name:
            self.cfg.save_path = self.cfg.local_root + 'models/Baseline_%s.pth'
        else:
            self.cfg.save_path = self.cfg.local_root + 'models/S2Site_%s.pth'
        self.cfg.initial_values_folder = self.cfg.local_root + 'models/initial_values/'

        # PPBS
        if self.cfg.mode == '_PPBS':
            self.cfg.label_path = self.cfg.local_root + 'datasets/PPBS/labels_%s.txt'
            self.cfg.dataset_wESM = self.cfg.dataset_root + 'datasets/pipelines/PPBS_ESM%s/PPBS_%s_pipeline_S2Site_aa-esm_atom-valency_frames-triplet_sidechain_Beff-500.data'
            self.cfg.dataset_path = self.cfg.dataset_root + 'datasets/pipelines/PPBS_%s_pipeline_S2Site_aa-sequence_atom-valency_frames-triplet_sidechain_Beff-500.data'
            self.cfg.dataset_train_path = self.cfg.dataset_root + 'datasets/pipelines/PPBS_ESM%s/train/PPBS_%s_%i.data'

            self.cfg.dataset_names = [
                'train',                #0
                'validation_70',        #1
                'validation_homology',  #2
                'validation_topology',  #3
                'validation_none',      #4
                'test_70',              #5
                'test_homology',        #6
                'test_topology',        #7
                'test_none'             #8
            ]

            self.cfg.dataset_titles = [
                'Train',
                'Validation (70\%)',
                'Validation (Homology)',
                'Validation (Topology)',
                'Validation (None)',
                'Test (70\%)',
                'Test (Homology)',
                'Test (Topology)',
                'Test (None)'
            ]

            self.cfg.dataset_table = self.cfg.local_root + 'datasets/PPBS/table.csv'

        elif 'BCE' in self.cfg.mode:
            # BCE
            self.cfg.label_path = self.cfg.local_root + 'datasets/BCE/labels_%s.txt'
            self.cfg.dataset_wESM = self.cfg.dataset_root + 'datasets/pipelines/BCE_ESM%s/BCE_%s_pipeline_S2Site_aa-esm_atom-valency_frames-triplet_sidechain_Beff-500.data'
            self.cfg.dataset_path = self.cfg.dataset_root + 'datasets/pipelines/BCE_%s_pipeline_S2Site_aa-sequence_atom-valency_frames-triplet_sidechain_Beff-500.data' 

            self.cfg.dataset_names = [
                'fold1', # 0
                'fold2', # 1
                'fold3', # 2
                'fold4', # 3
                'fold5', # 4
                ]

            self.cfg.dataset_titles = [
                'Fold 1', # 0
                'Fold 2', # 1
                'Fold 3', # 2
                'Fold 4', # 3
                'Fold 5'  # 4
            ]

            self.cfg.dataset_table = self.cfg.local_root + 'datasets/BCE/table.csv'
        
        elif 'PPeBS' in self.cfg.mode:
            # Peptide
            self.cfg.label_path = self.cfg.local_root + 'datasets/PPeBS/labels_%s.txt'
            self.cfg.dataset_wESM = self.cfg.dataset_root + 'datasets/pipelines/PPeBS_ESM%s/PPeBS_%s_pipeline_S2Site_aa-esm_atom-valency_frames-triplet_sidechain_Beff-500.data'
            self.cfg.dataset_path = self.cfg.dataset_root + 'datasets/pipelines/PPeBS_%s_pipeline_S2Site_aa-sequence_atom-valency_frames-triplet_sidechain_Beff-500.data' 
                                                           
            self.cfg.dataset_names = [
                'TR1038', # 0
                'TE125' # 1
                ]

            self.cfg.dataset_titles = [
                'Train', # 0
                'Test' # 1
                ]

            self.cfg.dataset_table = ''
        
        
        if self.cfg.dataset_table == '':
            self.cfg.reweight = False
        
        if not self.cfg.reweight:
            self.cfg.dataset_table = None
            
    def use_reweight(self, reweight=True):
        self.cfg.reweight = reweight
        self.reset_path()
    
    def use_reg(self, reg=True):
        self.cfg.reg = reg
        
    def use_atom(self, with_atom=True):
        self.cfg.with_atom = with_atom
        if with_atom:
            self.cfg.dense_pooling = 64
        else:
            self.cfg.dense_pooling = 0
        
    def use_mask(self, mask=True):
        self.cfg.mask = mask
        
    def downloadPDB(self, download=False):
        self.cfg.download_noFound = download
    
    def get_setting(self):
        return self.cfg
    
    def set_setting(self, new_config):
        self.cfg = new_config
        
    
    def reset_environment(self, num_of_threads=8, set_cuda=True, device=None, worker=8, val=True, val_over_=1):
        self.cfg.num_of_threads = num_of_threads
        self.cfg.set_cuda = set_cuda
        self.cfg.workers = worker
        self.cfg.val = val
        self.cfg.val_over_ = val_over_

        torch.set_num_threads(num_of_threads)
        if self.cfg.set_cuda and device is not None and device != 'cpu':
            self.cfg.device = f'cuda:{device}'
        else:
            self.cfg.device = 'cpu'


# for re-build training model
# set seed
def set_seed(seed):
    torch.manual_seed(seed) # cpu seed
    torch.cuda.manual_seed(seed) # gpu seed
    torch.backends.cudnn.deterministic = True # cudnn
    np.random.seed(seed) # numpy
    random.seed(seed) # random and transform

