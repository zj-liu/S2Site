from functools import partial
import numpy as np
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn
import torch.nn.functional as F


from networks.other_components import *
import networks.computations as computations



def embedding_initializer(input_feats, output_embed_dim):
    embed_attr = nn.Embedding(input_feats, output_embed_dim, padding_idx=0)
    embed_attr.weight.requires_grad = False
    if input_feats == output_embed_dim + 1:
        embed_attr.weight[1:] = (torch.eye(output_embed_dim) * np.sqrt(input_feats-1)).float()
    else:
        embed_attr.weight[1:, :] /= torch.sqrt((embed_attr.weight[1:, :]**2).mean(0))[np.newaxis, :]
        embed_attr.weight = ConstraintParameter(embed_attr.weight)
        embed_attr.weight.add_constraints(constraint=FixedNorm(axis=0, value=np.sqrt(input_feats-1)))
        embed_attr.weight = embed_attr.weight.apply_constraint()
    return embed_attr


class Attr_Embedding(nn.Module):
    def __init__(self, input_size=20, output_size=32, bias=True, mask=False):
        super(Attr_Embedding, self).__init__()
        self.mask = mask
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=bias)
        if self.mask:
            self.mbn = MaskedBatchNorm1d(output_size)
        else:
            self.bn = nn.BatchNorm1d(output_size)
        
        self.act = nn.ReLU()

    def forward(self, x, mask=None):
        y = self.linear(x)
        y = y * mask
        y = self.mbn(y.transpose(1,2), mask.transpose(1,2)).transpose(1,2)
        return self.act(y), mask


class GaussianKernel(nn.Module):
    def __init__(self, N, initial_values, covariance_type='full', ndim=3, feat_dim=3, eps=1e-1, mask=False):
        super(GaussianKernel, self).__init__()
        self.mask = mask
        
        self.eps = eps
        self.N = N
        self.initial_values = initial_values
        self.covariance_type = covariance_type

        self.nbatch_dim = ndim # len(input_shape) - 1
        self.d = feat_dim # input_shape[-1]

        self.center_shape = [self.d, self.N]
        assert self.covariance_type in ['diag', 'full']

        self.centers = nn.Parameter(torch.FloatTensor(self.initial_values[0]))

        if self.covariance_type == 'full':
            self.sqrt_precision_shape = [self.d, self.d, self.N]

            self.sqrt_precision = ConstraintParameter(torch.FloatTensor(self.initial_values[1]))
            self.sqrt_precision.add_constraints(constraint=ConstraintBetween(-1/self.eps,1/self.eps))
            self.sqrt_precision = self.sqrt_precision.apply_constraint()
            
        elif self.covariance_type == 'diag':
            self.width_shape = [self.d, self.N]

            self.widths = ConstraintParameter(torch.FloatTensor(self.initial_values[1]))
            self.widths.add_constraints(constraint=NonNeg)
            self.widths = self.widths.apply_constraint()
            

    def forward(self, inputs, mask):
        if self.covariance_type == 'full':
            # B X L X K X d X N
            intermediate = torch.unsqueeze(inputs, dim=-1) - torch.reshape(self.centers, [1 for _ in range(self.nbatch_dim)] + self.center_shape)  
            intermediate2 = torch.sum(
                torch.unsqueeze(intermediate, dim=-3) *
                torch.unsqueeze(self.sqrt_precision, dim=0),
                dim=-2)

            activity = torch.exp(- 0.5 * torch.sum(intermediate2**2,dim=-2))
            return activity, mask
        
        elif self.covariance_type == 'diag':
            activity = torch.exp(- 0.5 * torch.sum(
                (
                    (
                        torch.unsqueeze(inputs, dim=-1)
                        - torch.reshape(self.centers,
                                    [1 for _ in range(self.nbatch_dim)] + self.center_shape)
                    ) / torch.reshape(self.eps + self.widths, [1 for _ in range(self.nbatch_dim)] + self.width_shape)
                )**2, dim=-2))
            return activity, mask
        
        return None


class Outer_Product(nn.Module):
    def __init__(self, n_filters, coord_feat_dim=32, attr_feat_dim=12, use_single1=True, use_single2=True, 
                    use_bias=True, non_negative=False, unitnorm=False, fixednorm=None,
                    symmetric=False, diagonal = False, non_negative_initial=False,
                    kernel_regularizer=None, single1_regularizer=None, single2_regularizer=None, 
                    sum_axis=None, mask=False):
        super(Outer_Product, self).__init__()
        self.mask = mask

        self.n_filters = n_filters
        self.use_single1 = use_single1
        self.use_single2 = use_single2
        self.use_bias = use_bias
        self.non_negative = non_negative
        self.kernel_regularizer = kernel_regularizer
        self.single1_regularizer = single1_regularizer
        self.single2_regularizer = single2_regularizer

        if unitnorm:  # for retro-compatibility...
            fixednorm = 1.0
        self.fixednorm = fixednorm
        self.symmetric = symmetric
        self.diagonal = diagonal
        self.sum_axis = sum_axis
        self.non_negative_initial = non_negative_initial


        if self.non_negative:
            constraint = NonNeg
        else:
            constraint = None


        if self.fixednorm is not None:
            constraint_kernel = FixedNorm(value=self.fixednorm, axis=[0, 1])
        else:
            constraint_kernel = constraint


        if self.symmetric:
            constraint_kernel = Symmetric


        self.n1 = coord_feat_dim
        self.n2 = attr_feat_dim


        if self.fixednorm is not None:
            stddev = self.fixednorm / np.sqrt(self.n1 * self.n2)
        else:
            if self.diagonal:
                stddev = 1.0 / np.sqrt(self.n1)
            else:
                stddev = 1.0 / np.sqrt(self.n1 * self.n2)


        if self.non_negative_initial:
            initializer = partial(nn.init.uniform_, mean=0, std=stddev * np.sqrt(3)) # such that < w^2 > = stddev exactly.
        else:
            initializer = partial(nn.init.trunc_normal_, mean=0, std=stddev)
        
        
        if self.diagonal:
            weight = torch.empty(self.n1, self.n_filters)
            weight = initializer(weight)
            self.kernel12 = nn.Parameter(weight)
        else:
            weight = torch.empty(self.n1, self.n2, self.n_filters)
            weight = nn.init.trunc_normal_(weight, mean=0, std=stddev)
            self.kernel12 = ConstraintParameter(weight)
            self.kernel12.add_constraints(constraint=constraint_kernel, regularizer=self.kernel_regularizer)
            self.kernel12 = self.kernel12.apply_constraint()


        if self.use_single1:
            stddev = 1.0 / np.sqrt(self.n1)
            if self.non_negative_initial:
                initializer = partial(nn.init.uniform_, mean=0, std=stddev * np.sqrt(3)) # such that < w^2 > = stddev exactly.

            else:
                initializer = partial(nn.init.trunc_normal_, mean=0, std=stddev)

            weight = torch.empty(self.n1, self.n_filters)
            weight = initializer(weight)
            self.kernel1 = ConstraintParameter(weight)
            self.kernel1.add_constraints(constraint=constraint, regularizer=self.single1_regularizer)
            self.kernel1 = self.kernel1.apply_constraint()


        if self.use_single2:
            stddev = 1.0 / np.sqrt(self.n2)
            if self.non_negative_initial:
                initializer = partial(nn.init.uniform_, mean=0, std=stddev * np.sqrt(3)) # such that < w^2 > = stddev exactly.
            else:
                initializer = partial(nn.init.trunc_normal_, mean=0, std=stddev)

            if self.symmetric:
                self.kernel2 = self.kernel1
            else:
                weight = torch.empty(self.n2, self.n_filters)
                weight = initializer(weight)
                self.kernel2 = ConstraintParameter(weight)
                self.kernel2.add_constraints(constraint=constraint, regularizer=self.single2_regularizer)
                self.kernel2 = self.kernel12.apply_constraint()


        if self.use_bias:
            weight = torch.empty(self.n_filters)
            self.bias = nn.Parameter(nn.init.zeros_(weight))    


    def forward(self, inputs, mask):
        # inputs = [coord, attr]
        first_input = inputs[0]
        second_input = inputs[1]
        bias_shape = [1 for _ in first_input.shape[:-1]] + [self.n_filters]

        if self.sum_axis is not None:
            del bias_shape[self.sum_axis]
        
        if self.diagonal:
            activity = torch.matmul(first_input * second_input, self.kernel12)
        else:
            # B, L, Ngaussian=32, Nattribute=12 if atom
            if self.sum_axis is not None:
                outer_product = torch.sum(torch.unsqueeze(
                    first_input, axis=-1) * torch.unsqueeze(second_input, dim=-2), dim=self.sum_axis)
            else:
                outer_product = torch.unsqueeze(
                    first_input, dim=-1) * torch.unsqueeze(second_input, dim=-2)
            
            # kernel12 = Ngaussian, Nattribute, Nfilter=128
            activity = torch.tensordot(outer_product, self.kernel12, [[-2, -1], [0, 1]])

        if self.use_single1:
            if self.sum_axis is not None:
                activity = activity + torch.matmul(torch.sum(first_input, dim=self.sum_axis), self.kernel1)
            else:
                activity = activity + torch.matmul(first_input, self.kernel1)
        if self.use_single2:
            if self.sum_axis is not None:
                activity = activity + torch.matmul(torch.sum(second_input, axis=self.sum_axis), self.kernel2)
            else:
                activity = activity + torch.matmul(second_input, self.kernel2)

        if self.use_bias:
            activity = activity + torch.reshape(self.bias, bias_shape)
        
        return activity, self.compute_mask(mask)


    def compute_mask(self, mask=None):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if self.sum_axis is not None:
            return mask[0][..., 0]
        else:
            return mask[0]


    def get_config(self):
        class_info = {'n_filters': self.n_filters,
                    'use_single1': self.use_single1,
                    'use_single2': self.use_single2,
                    'use_bias': self.use_bias,
                    'non_negative': self.non_negative,
                    'fixednorm': self.fixednorm,
                    'symmetric':self.symmetric,
                    'diagonal':self.diagonal,
                    'sum_axis': self.sum_axis,
                    }
        return class_info


class Neighborhood_Embedding(nn.Module):
    def __init__(self, config, info, which_gaussian='atom', Kmax=16, nrotations=1, 
                    attr_feat_dim=12, index_distance_max=8, input_len=2, mask=False):
        super(Neighborhood_Embedding, self).__init__()
        self.mask = mask
        self.K = Kmax
        self.gaussian_kernel = which_gaussian
        if which_gaussian == 'dense_graph':
            which_gaussian = 'graph'
        self.nrotations = nrotations

        self.nc = computations.Neighborhood_Computation(K_neighbor=Kmax, index_distance_max=index_distance_max, 
                                                            coordinates=config['coordinates_'+ which_gaussian], 
                                                            nrotations=self.nrotations, input_len=input_len, mask=self.mask, device=config.device)

        l1 = config['l1_' + which_gaussian]
        l12 = config['l12_' + which_gaussian]
        l12group = config['l12group_' + which_gaussian]

        nfilters = config['nfilters_'+ which_gaussian]
        self.Ngaussians = config['N_' + which_gaussian]

        if self.gaussian_kernel != 'dense_graph':
            self.gaussian_kernel = 'GaussianKernel_' + self.gaussian_kernel

        initial_gaussian_values = info['initial_values'][self.gaussian_kernel]
        # self.gk_coord = GaussianKernel(N=self.Ngaussians, initial_values=initial_gaussian_values, mask=self.mask)
        self.coord_embed = GaussianKernel(N=self.Ngaussians, initial_values=initial_gaussian_values, mask=self.mask)

        self.kernel_regularizer = None
        self.single1_regularizer = None
        self.fixednorm = None
        # Apply Spatio-chemical filters.
        if l1 > 0:
            self.kernel_regularizer = partial(l1_regularization, l1=l1)
            self.single1_regularizer = self.kernel_regularizer
            self.fixednorm = np.sqrt(self.Ngaussians / Kmax)
        elif l12 > 0:
            self.kernel_regularizer = partial(l12_regularization, l12=l12, ndims=3)
            self.single1_regularizer = partial(l12_regularization, l12=l12, ndims=2)
            self.fixednorm = np.sqrt(self.Ngaussians / Kmax)
        elif l12group >0:
            self.kernel_regularizer = partial(l12group_regularization, l12group=l12group, ndims=3)
            self.single1_regularizer = partial(l12group_regularization, l12group=l12group, ndims=2)
            self.fixednorm = np.sqrt(self.Ngaussians / Kmax)


        self.outerp = Outer_Product(n_filters=nfilters, use_single1=True, use_single2=False, use_bias=False, 
                            kernel_regularizer=self.kernel_regularizer, single1_regularizer=self.single1_regularizer, 
                            fixednorm=self.fixednorm, non_negative=False, non_negative_initial=False, sum_axis=2, 
                            coord_feat_dim=self.Ngaussians, attr_feat_dim=attr_feat_dim, mask=self.mask)

        if self.mask:
            self.mbn = MaskedBatchNorm1d(nfilters)
        else:
            self.bn = nn.BatchNorm1d(nfilters)
        self.act = nn.ReLU()

        # Note: Default setting does not require nrotations > 1
        if self.nrotations > 1:
            print('Setting - nrotation is not equal to 1! Additional functions required for NEM.')
            return
    
    def reset_device(self, device):
        self.nc.reset_device(device)

    def forward(self, inputs, mask):
        # inputs = [frame_feats, attr]
        
        # [(B, L=9216, K=16, 3), (B, L=9216, K=16, 12)]
        out = self.nc(inputs, mask)
        coord_feats, attr_feats = out[0]
        mcoord, mattr = out[1]
            
        # Gaussian embedding of local coordinates.
        # B, L, K, Ngaussians=32
        coord_feats, mcoord = self.coord_embed(coord_feats, mcoord)
        coord_feats = coord_feats * mcoord
        
        # B, L, Embed=128
        spatiochemical_filters_input, mspatiochemical = self.outerp([coord_feats, attr_feats], mask=[mcoord, mattr])
        spatiochemical_filters_input = spatiochemical_filters_input * mspatiochemical

        out = self.mbn(spatiochemical_filters_input.transpose(1, 2), mspatiochemical.transpose(1,2)).transpose(1, 2)
        return self.act(out), mspatiochemical


