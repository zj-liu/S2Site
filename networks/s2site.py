import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from functools import partial

import torch
import torch.nn as nn

from utilities import io_utils, wrappers
from networks import embeddings, computations, attention

from networks.other_components import *


class Block_Atom(nn.Module):
    def __init__(self, config, additional_info):
        super().__init__()
        self.mask = config.mask

        self.nembedding_attr = config.nembedding_atom
        self.ncategory = config.nfeatures_atom
        self.nembedding_neighborhood = config.nfilters_atom

        self.embed_attr = embeddings.embedding_initializer(self.ncategory+1, self.nembedding_attr)
        self.frame_builder = computations.Frame_Builder(order=additional_info['order_atom'], dipole=additional_info['dipole_atom'], mask=self.mask, device=config.device)

        self.nem = embeddings.Neighborhood_Embedding(config=config, info=additional_info, Kmax=config.K_atom, mask=self.mask)

        self.pooling_attention = nn.Linear(in_features=self.nembedding_neighborhood, out_features=config.nattentionheads_pooling, bias=False)

        # Pool at amino acid level by attention-pooling.
        if config.dense_pooling is None:
            config.dense_pooling = config.nfilters_atom

        self.pooling_features = nn.Linear(in_features=self.nembedding_neighborhood, out_features=config.dense_pooling, bias=False)

        if config.l12_pool > 0:
            regular = partial(l12_regularization, l12=config.l12_pool)

            self.pooling_attention.weight = ConstraintParameter(self.pooling_attention.weight.data.zero_())
            self.pooling_attention.weight.add_constraints(regularizer=regular)

            self.pooling_features.weight = ConstraintParameter(self.pooling_features.weight)
            self.pooling_features.weight.add_constraints(regularizer=regular, constraint=FixedNorm(1.0, axis=0))
            self.pooling_features.weight = self.pooling_features.weight.apply_constraint()
        else:
            self.pooling_attention.weight = nn.Parameter(nn.init.zeros_(self.pooling_attention.weight))
        

        self.nc = computations.Neighborhood_Computation(K_neighbor=14, coordinates=['index_distance'], input_len=4,
                                                            self_neighborhood=False, index_distance_max=1, device=config.device)

        self.pool_mask = lambda x : 1 - x

        self.pooling = attention.Attention_Layer(self_attention=False, beta=False)

        if self.mask:
            self.mbn = MaskedBatchNorm1d(config.dense_pooling)
        else:
            self.bn = nn.BatchNorm1d(config.dense_pooling)
        self.act = nn.ReLU()
        
    def reset_device(self, device):
        self.device = device
        self.to(device)
        self.nem.reset_device(device)
        self.frame_builder.reset_device(device)
        self.nc.reset_device(device)

    def forward(self, atom_feats, sequence_indices_aa, mask_atom, mseq_aa):
        frame_indices_atom, attr_embedding_atom, sequence_indices_atom, point_clouds_atom = atom_feats
        frame_indices_atom = frame_indices_atom.to(self.device)
        attr_embedding_atom = attr_embedding_atom.to(self.device)
        sequence_indices_atom = sequence_indices_atom.to(self.device)
        point_clouds_atom = point_clouds_atom.to(self.device)
        sequence_indices_aa = sequence_indices_aa.to(self.device)
        
        # attr_atom is one-hot encoded as 12 categories [B, L=9216, embeddings=12] 
        attr_embedding_atom = self.embed_attr(attr_embedding_atom)
        
        mseq_aa.to(self.device)
        
        mframe, _, mseq, mpc = mask_atom
        mframe = mframe.to(self.device)
        mseq = mseq.to(self.device)
        mpc = mpc.to(self.device)
        
        mattr = (1 * torch.any(attr_embedding_atom, dim=-1, keepdim=True)).to(self.device)

        # get atoms' frames [B, L=9216, 4, 3]
        frame_feats_atom, mframe = self.frame_builder(frame_indices_atom, point_clouds_atom, mask=[mframe, mpc])
        frame_feats_atom = frame_feats_atom * mframe

        # atomic neighborhood embeddings [B, L, nembedding_neighborhood]
        y, mask_y = self.nem([frame_feats_atom, attr_embedding_atom], mask=[mframe, mattr])
        y = y * mask_y

        # Compute attention coefficients (before softmax) for each atom. [B, L, nattentionheads_pooling=64]
        pooling_attention = self.pooling_attention(y)
        pooling_attention = pooling_attention * mask_y

        # Compute output coefficients (before averaging) for each atom. [B, L, dense_pooling = 64 (specified) or nfilter_atom]
        pooling_features = self.pooling_features(y)
        pooling_features = pooling_features * mask_y

        # Build a binary bipartite graph from amino acid to atoms.
        # For each amino acid we look for the 14 closest atoms in terms of sequence distance.
        # indice_diff = [B, Laa=1024, Kmax2=14, bool_dim=1]
        # pooling_attention_local =[B, Laa=1024, Kmax2=14, nembedding=64]
        # pooling_features_local = [B, Laa=1024, Kmax2=14, dense_pooling=64]
        out = self.nc([sequence_indices_aa, sequence_indices_atom, pooling_attention, pooling_features], mask=[mseq_aa, mseq, mask_y, mask_y])
        indice_diff, pooling_attention_local, pooling_features_local = out[0]
        mask_pm, mask_pal, mask_pfl = out[1]

        # Note - is mask really required...?
        # Reverse the pooling mask such that M_{ij} =1 iff atom j belongs to amino acid i.
        indice_diff = self.pool_mask(indice_diff)

        # [B, Laa, dense=64], [B, Laa, nembedding=64]
        attrs, masks = self.pooling([pooling_attention_local, pooling_features_local, indice_diff], mask=[mask_pal, mask_pfl, mask_pm])
        SCAN_filters_atom_aggregated_input = attrs[0]
        mask_scan = masks[0]
        if mask_scan is not None:
            SCAN_filters_atom_aggregated_input = SCAN_filters_atom_aggregated_input * mask_scan
            SCAN_filters_atom_aggregated_input = self.mbn(SCAN_filters_atom_aggregated_input.transpose(1,2), mask_scan.transpose(1, 2)).transpose(1, 2)
        else:
            SCAN_filters_atom_aggregated_input = self.mbn(SCAN_filters_atom_aggregated_input.transpose(1, 2), mask_scan).transpose(1, 2)
        return self.act(SCAN_filters_atom_aggregated_input), mask_scan
    

class S2Site(nn.Module):
    def __init__(self, config, info_from_init):
        super(S2Site, self).__init__()

        self.config = config
        self.mask = config.mask
        self.return_intermediate()
        # atom part
        self.with_atom = config.with_atom
        if self.with_atom:
            self.block_atom = Block_Atom(config=config, additional_info=info_from_init)

        if config.use_esm:
            self.dropout = nn.Dropout(config.dropout)
        # aa part
        self.nam_aa = embeddings.Attr_Embedding(input_size=config.nfeatures_aa, output_size=config.nembedding_aa, bias=False, mask=self.mask)
        self.fb = computations.Frame_Builder(order=info_from_init['order_aa'], dipole=info_from_init['dipole_aa'], mask=self.mask, device=config.device)


        self.nem_aa = embeddings.Neighborhood_Embedding(config=config, info=info_from_init, which_gaussian='aa', Kmax=config.K_aa, mask=self.mask, attr_feat_dim=config.nembedding_aa+config.dense_pooling)                                                                 
        
        self.nams = []
        prev_size = config.nfilters_aa
        for nfilter in config.filter_MLP:
            self.nams.append(embeddings.Attr_Embedding(prev_size, nfilter, bias=False, mask=self.mask))
            prev_size = nfilter
        self.nams = nn.ModuleList(self.nams)

        # Neighborhood attention part
        self.beta = nn.Linear(prev_size, config.nembedding_graph*config.nattentionheads_graph)
        self.beta.weight.data.zero_()
        self.beta.bias.data.zero_()
        self.beta.bias.data += 1


        self.relu = nn.ReLU()


        self.self_attention = nn.Linear(prev_size, config.nembedding_graph*config.nattentionheads_graph)
        self.self_attention.weight.data.zero_()
        self.self_attention.bias.data.zero_()


        self.cross_attention = nn.Linear(prev_size, config.nembedding_graph*config.nattentionheads_graph, bias=False)
        self.cross_attention.weight.data.zero_()

        
        self.node_features = nn.Linear(prev_size, config.nattentionheads_graph*config.nfilters_graph)
        self.node_features_activation = None
        if config.output == 'classification' and config.nfilters_graph > 2:
            self.node_features_activation = 'relu'


        self.nc = computations.Neighborhood_Computation(K_neighbor=config.K_graph, coordinates=config.coordinates_graph, 
                                                            index_distance_max=config.index_distance_max_graph, input_len=4, mask=self.mask, device=config.device)

        self.gk = embeddings.GaussianKernel(N=config.N_graph, initial_values=info_from_init['initial_values']['GaussianKernel_graph'],
                                                covariance_type=config.covariance_type_graph, feat_dim=5, mask=self.mask)

        self.l1 = nn.Linear(config.N_graph, config.nembedding_graph, bias=False)
        self.l1.weight = nn.Parameter(torch.FloatTensor(info_from_init['initial_values']['dense_graph'][0].transpose()))
        
        self.attent = attention.Attention_Layer()

        self.classifier_layer = None
        self.output_act = None
        if config.output == 'classification':
            if config.nattentionheads_graph * config.nfilters_graph > 2:
                self.classifier_layer = nn.Linear(config.nembedding_graph, 2)
                self.output_act = nn.Softmax(dim=-1)
            else:
                self.output_act = nn.Softmax(dim=-1)
                
        self.reset_device(config.device)
                
            
    def reset_device(self, device):
        self.device = device
        self.to(device)
        self.fb.reset_device(device)
        self.nc.reset_device(device)
        self.nem_aa.reset_device(device)
        
        if self.with_atom:
            self.block_atom.reset_device(device)
        return None
    
    def return_intermediate(self, return_mid=False):
        self.return_mid = return_mid

    def forward(self, x, mask):
        feats_aa, feats_atom = x[:4], x[4:]
        frame_indices_aa, attr_aa, sequence_indices_aa, point_clouds_aa = feats_aa
        frame_indices_aa = frame_indices_aa.to(self.device)
        attr_aa = attr_aa.to(self.device)
        sequence_indices_aa = sequence_indices_aa.to(self.device)
        point_clouds_aa = point_clouds_aa.to(self.device)

        
        mframe_aa, mattr_aa, mseq_aa, mpc_aa = mask[:4]
        mframe_aa = mframe_aa.to(self.device)
        mattr_aa = mattr_aa.to(self.device)
        mseq_aa = mseq_aa.to(self.device)
        mpc_aa = mpc_aa.to(self.device)
        
        if attr_aa.shape[-1] != 20:
            attr_aa = self.dropout(attr_aa)

        attr_embedding, mattr_aa = self.nam_aa(attr_aa, mattr_aa)

        if self.with_atom:
            # B, Laa, 64
            pooled_feats_atom, mask_ = self.block_atom(atom_feats=feats_atom, sequence_indices_aa=sequence_indices_aa, mask_atom=mask[4:], mseq_aa=mseq_aa)
            pooled_feats_atom.to(self.device)
            if mask_ is not None:
                mattr_aa = 1 * torch.any(mattr_aa | mask_.to(self.device), dim=-1, keepdim=True)
            # B, Laa, 96=32+64
            attr_embed = torch.cat((attr_embedding, pooled_feats_atom), dim=-1) * mattr_aa
        
        # B, Laa, 4, 3
        frame_feats_aa, mframe_aa = self.fb(frame_indices_aa, point_clouds_aa, mask=[mframe_aa, mpc_aa])
        frame_feats_aa *= mframe_aa
        
        if self.with_atom:
            # B, Laa, 128
            y, mask_y = self.nem_aa([frame_feats_aa, attr_embed], mask=[mframe_aa, mattr_aa])
        else:
            y, mask_y = self.nem_aa([frame_feats_aa, attr_embedding], mask=[mframe_aa, mattr_aa])
        y = y * mask_y

        # B, Laa, 32 by filter_MLP=[32]
        for i in range(len(self.config.filter_MLP)):
            y, mask_y = self.nams[i](y, mask_y)
            y = y * mask_y
            
        # Final graph attention layer. 
        # Propagates label information from "hotspots" to passengers to obtain spatially consistent labels.
        
        # B, Laa, 1
        beta = self.beta(y) * mask_y
        beta = self.relu(beta)
        self_attention = self.self_attention(y)
        self_attention = self_attention * mask_y
        cross_attention = self.cross_attention(y)
        cross_attention = cross_attention * mask_y

        # B, Laa, 2
        node_feats = self.node_features(y)
        node_feats = node_feats * mask_y
        if self.node_features_activation is not None:
            node_feats = self.relu(node_feats)

        # [B, Laa, K_graph=32, 5], [B, Laa, 32, 1], [B, Laa, 32, 2]
        out = self.nc([frame_feats_aa, sequence_indices_aa, cross_attention, node_feats], mask=[mframe_aa, mseq_aa, mask_y, mask_y])
        graph_weights, attention_local, node_features_local = out[0]
        mask_gw, mask_al, mask_nfl = out[1]

        # [B, Laa, 32, 32]
        graph_weights, mask_gw = self.gk(graph_weights, mask_gw)
        graph_weights = graph_weights * mask_gw

        # [B, Laa, 32, 1]
        graph_weights = self.l1(graph_weights)
        graph_weights = graph_weights * mask_gw
        
        # beta=constrast_coefficient, self_attention=self_attention, 
        # attention_local=cross_attention, node_features_local=output_features, graph_weights=learnt_graph_edges
        # [B, Laa, 2], [B, Laa, 32, 1]
        attent_output, attent_masks = self.attent([beta, self_attention, attention_local, node_features_local, graph_weights], 
                                                    mask=[mask_y, mask_y, mask_al, mask_nfl, mask_gw])

        graph_attention_output = attent_output[0] 
        mask_gao = attent_masks[0]
        if mask_gao is not None:
            graph_attention_output = graph_attention_output * mask_gao
        else:
            mask_gao = torch.ones(mask_gao.shape)

        if self.classifier_layer is not None:
            graph_attention_output = self.classifier_layer(graph_attention_output)
            graph_attention_output = graph_attention_output * mask_gao
        
        if self.return_mid:
            return graph_attention_output, attr_embedding, y
        return graph_attention_output


def initial_S2Site(config, get_model=True):
    Lmax_atom = 9 * config.Lmax_aa
    if config.frame_aa == 'triplet_backbone':
        Lmax_aa_points = config.Lmax_aa + 2
    elif config.frame_aa in ['triplet_sidechain','triplet_cbeta']:
        Lmax_aa_points = 2 * config.Lmax_aa + 1
    elif config.frame_aa == 'quadruplet':
        Lmax_aa_points = 2 * config.Lmax_aa + 2
    Lmax_atom_points = 11 * config.Lmax_aa


    initial_values = {'GaussianKernel_aa': None, 'GaussianKernel_atom': None, 'GaussianKernel_graph': None,'dense_graph': None}

    if config.with_atom:
        input_type = ['triplets', 'attributes', 'indices', 'points', 'triplets', 'attributes', 'indices', 'points']
        Lmaxs = [config.Lmax_aa, config.Lmax_aa, config.Lmax_aa, Lmax_aa_points, 
                Lmax_atom, Lmax_atom, Lmax_atom, Lmax_atom_points]
    else:
        input_type = ['triplets', 'attributes', 'indices', 'points']
        Lmaxs = [config.Lmax_aa, config.Lmax_aa, config.Lmax_aa, Lmax_aa_points]

    if config.frame_aa == 'triplet_backbone':
        order_aa = '3'
        dipole_aa = False
    elif config.frame_aa in ['triplet_sidechain', 'triplet_cbeta']:
        order_aa = '2'
        dipole_aa = False
    elif config.frame_aa == 'quadruplet':
        order_aa = '3'
        dipole_aa = True
    else:
        print('Incorrecte frame_aa')
        return

    order_atom = '2'
    dipole_atom = False
    frame_atom = 'covalent'

    c_aa = ''.join([c[0] for c in config.coordinates_aa])
    location_aa = config.initial_values_folder + 'initial_GaussianKernel_aa_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_nrotations_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
        config.N_aa, config.K_aa, config.Dmax_aa, config.frame_aa, c_aa, config.nrotations, config.covariance_type_aa, order_aa, dipole_aa)
    #print('location_aa:', location_aa)

    try:
        assert config.fresh_initial_values == False
        initial_values['GaussianKernel_aa'] = io_utils.load_pickle(location_aa)['GaussianKernel_aa']
    except:
        print("Can't find GaussianKernel_aa")

    if config.with_atom:
        #print(
        #    'Initializing the Gaussian kernels for the atomic neighborhood (takes a few minutes to do it robustly, be patient!). Reduce n_init from 10 to 1 if speed needed')
        c_atom = ''.join([c[0] for c in config.coordinates_atom])
        location_atom = config.initial_values_folder + 'initial_GaussianKernel_atom_N_%s_Kmax_%s_Dmax_%s_frames_%s_coords_%s_nrotations_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
            config.N_atom, config.K_atom, config.Dmax_atom, frame_atom, c_atom, 1, config.covariance_type_atom, order_atom, dipole_atom)
        #print('location_atom:', location_atom)

        try:
            assert config.fresh_initial_values == False
            initial_values['GaussianKernel_atom'] = io_utils.load_pickle(location_atom)['GaussianKernel_atom']
        except:
            print("Can't find GaussianKernel_atom")

    c_graph = ''.join([c[0] for c in config.coordinates_graph])
    # print(config.initial_values_folder)
    location_graph = config.initial_values_folder + 'initial_GaussianKernel_graph_N_%s_%s_Kmax_%s_Dmax_%s_coords_%s_indexmax_%s_cov_%s_tripletsorder_%s_dipole_%s.data' % (
        config.N_graph, config.nembedding_graph, config.K_graph, config.Dmax_graph, c_graph, config.index_distance_max_graph, config.covariance_type_graph,
        order_aa, dipole_aa)
    #print('location_graph:', location_graph)

    try:
        assert config.fresh_initial_values == False
        initial_values_graph = io_utils.load_pickle(location_graph)
    except:
        print("Can't find GaussianKernel_graph")
        
    initial_values['GaussianKernel_graph'] = initial_values_graph['graph_embedding_GaussianKernel']
    initial_values['dense_graph'] = initial_values_graph['graph_embedding_dense']

    info = {'Lmax_atom' : Lmax_atom,  'max_aa_points' : Lmax_aa_points,
            'Lmax_atom_points' : Lmax_atom_points,
            'initial_values' : initial_values,
            'input_type' : input_type,
            'Lmaxs' : Lmaxs,
            'multi_inputs' :True,
            'multi_outputs':False, 
            'order_aa': order_aa,
            'dipole_aa': dipole_aa,
            'order_atom': order_atom,
            'dipole_atom': dipole_atom,
            'frame_atom': frame_atom}

    wrapper = wrappers.grouped_Predictor_wrapper(input_type=input_type, Lmaxs=Lmaxs)
    if get_model:
        model = S2Site(config=config,info_from_init=info)
        return model, wrapper, info
    return wrapper, info
