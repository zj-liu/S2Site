import numpy as np

import torch
import torch.nn as nn


''' Regularization Part '''
def l1_regularization(W, l1):
    return l1 * torch.sum(torch.abs(W))


def l12_regularization(W, l12, ndims=2, order='gaussian_feature_filter'):
    if order == 'filter_gaussian_feature':  # Order of the tensor indices.
        if ndims == 2: # For gaussian-dependent bias
            return l12 / 2 * float(W.shape[1]) * torch.sum(torch.square(torch.mean(torch.abs(W), dim=1)))
        elif ndims == 3:
            return l12 / 2 * float(W.shape[1] * W.shape[2]) * torch.sum(
                torch.square(torch.mean(torch.abs(W), dim=(1, 2))))
    elif order == 'gaussian_feature_filter':  # Default order the tensor indices.
        if ndims == 2:
            return l12 / 2 * float(W.shape[0]) * torch.sum(torch.square(torch.mean(torch.abs(W), dim=0)))
        elif ndims == 3:
            return l12 / 2 * float(W.shape[0] * W.shape[1]) * torch.sum(
                torch.square(torch.mean(torch.abs(W), dim=(0, 1))))


def l12group_regularization(W, l12group, ndims=2, order='gaussian_feature_filter'):
    if ndims == 2: # For gaussian-dependent bias, Same as l12
        return l12_regularization(W, l12group, ndims=ndims, order=order)
    elif ndims == 3:
        if order == 'filter_gaussian_feature': # Order of the tensor indices.
            return l12group / 2 * float(W.shape[1] * W.shape[2]) * torch.sum(
                torch.square(torch.mean(torch.sqrt(torch.mean(torch.square(W), axis=-1)), axis=-1)))
    elif order == 'gaussian_feature_filter': # Order of the tensor indices.
        return l12group / 2 *  float(W.shape[0] * W.shape[1]) * torch.sum(
            torch.square(torch.mean(torch.sqrt(torch.mean(torch.square(W), axis=1)), axis=0)))






''' gather_nd solutions '''
# when batch_dim=1
def get_values_from_indices(params, indices):
    out = torch.zeros( (list( indices.size() ) + list(params.size()[len(indices.shape)-1:]) ), dtype=torch.float).to(params.device)
    for i in range(out.size(0)):
        out[i] = out[i] + params[i, indices[i]]
    return out


# when batch_dim=2
def get_values_from_indices_2d(params, indices):
    out = torch.zeros( (list( indices.size() ) + list(params.size()[len(indices.shape):]) ), dtype=torch.float).to(params.device)
    for i in range(out.size(0)):
        for j in range(out.size(1)):
            out[i, j] = out[i, j] + params[i, j, indices[i, j, :]]
    return out


def slice_computation(points, triplets, axis_i, axis_j=None, batch_dim=1):
    out = torch.zeros(( list(triplets.size()[:-1]) + list(points.size()[-1:]) ), dtype=torch.float).to(points.device)
    if batch_dim == 1:
        if axis_j is None:
            for i in range(triplets.size(0)):
                out[i] = out[i] + points[i, triplets[:, :, axis_i].long()][i]
        else:
            for i in range(triplets.size(0)):
                out[i] = out[i] + (points[i, triplets[:, :, axis_i].long()][i] - points[i, triplets[:, :, axis_j].long()][i])
    return out



''' Distance Functions'''
def distance(coordinates1, coordinates2, squared=False, ndims=3):
    D = ((torch.unsqueeze(coordinates1[...,0],axis=-1) - torch.unsqueeze(coordinates2[...,0],axis=-2) )**2).float().to(coordinates1.device)
    for n in range(1,ndims):
        D = D + ((torch.unsqueeze(coordinates1[..., n], axis=-1) - torch.unsqueeze(coordinates2[..., n], axis=-2)) ** 2).float()
    if not squared:
        D = torch.sqrt(D)
    return D


def euclidian_to_spherical(x, return_r=True, cut='2pi', eps=1e-8):
    r = torch.sqrt(torch.sum(x**2, dim=-1) )
    theta = torch.acos(x[...,-1]/(r+eps) )
    phi = torch.atan2( x[...,1],x[...,0]+eps)
    if cut == '2pi':
        phi = phi + torch.greater(0.,phi).float() * (2 * np.pi)
    if return_r:
        return torch.stack([r,theta,phi], dim=-1)
    else:
        return torch.stack([theta, phi], dim=-1)




''' Constraint '''
class ConstraintBetween():
    def __init__(self, minimum=-1, maximum=+1):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(self, w):
        return torch.clamp(w, self.minimum, self.maximum)


class FixedNorm():
    def __init__(self, value=1.0, axis=0):
        self.axis = axis
        self.value = value
        self.epsilon = 1e-07

    def __call__(self, w):
        return w * self.value / (
            self.epsilon + torch.sqrt(
                torch.sum(
                    torch.square(w), dim=self.axis, keepdims=True)))

    def get_config(self):
        return {'axis': self.axis, 'value': self.value}


def Symmetric(w):
    return (w + torch.transpose(w, [1, 0, 2]) ) / 2


def NonNeg(w):
    """Constraints the weights to be non-negative.
    """
    w = w * torch.greater_equal(w, 0.).float()
    return w








''' Constraint Parameter '''
class ConstraintParameter(nn.Parameter):
    constraint = None 
    regularizer = None

    def add_constraints(self, constraint=None, regularizer=None):
        self.constraint = constraint
        self.regularizer = regularizer

    def compute_regularization(self):
        return self.regularizer(self)
           
    def apply_constraint(self):
        if self.constraint is None:
           return self
        new_data = ConstraintParameter(self.constraint(self.data))
        new_data.add_constraints(constraint=self.constraint, regularizer=self.regularizer)
        return new_data









''' Masked BN 1d '''
class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - Input: :math:`(N, C, L)`
            - input_mask: (N, 1, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, input_mask=None):
        # print(self.running_mean.shape)
        # Calculate the masked mean and variance
        B, C, L = input.shape
        if input_mask is not None and input_mask.shape != (B, 1, L):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        if input_mask is not None:
            masked = input * input_mask
            n = input_mask.sum()
        else:
            masked = input
            n = B * L
        # Sum
        masked_sum = masked.sum([0, 2])
        # Divide by sum of mask
        current_mean = masked_sum / n
        # current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n  ###incorrect
        current_var = (masked ** 2).sum([0, 2]) / n - current_mean ** 2
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean[None, :, None]) / (torch.sqrt(self.running_var[None, :, None] + self.eps))
        else:
            normed = (masked - current_mean[None, :, None]) / (torch.sqrt(current_var[None, :, None] + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        # print(self.running_mean.shape)
        return normed
    


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, restore_best_weight=True, mode='min'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.restore_best_weight = restore_best_weight

        self.other_metric = None
        self.loop = -1
        self.mode = mode

        self.best_state = None

    def __call__(self, model, val_loss, loop, other_metric=None):
        if self.mode == 'min':
            score = -val_loss
        else:
            score = val_loss
            
        if self.best_score is None:
            self.best_score = score
            self.other_metric = other_metric
            self.loop = loop
            self.best_state = model.state_dict()
            self.save_checkpoint(val_loss)
            return True
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print('Restore best weight...')
            if self.restore_best_weight:
                model.load_state_dict(self.best_state)

            if self.counter >= self.patience:
                self.early_stop = True
                print('')
                print(f'EarlyStopping at {loop}! Val Loss:{self.val_loss_min}. Other Metrics:{self.other_metric} from epoch {self.loop}')
            return False
        else:
            self.best_score = score
            self.other_metric = other_metric
            self.loop = loop
            self.best_state = model.state_dict()
            self.save_checkpoint(val_loss)
            self.counter = 0
            return True

    #def save_checkpoint(self, val_loss, model, optimizer, path):
    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model checkpoint...')
        #torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, path)
        self.val_loss_min = val_loss






''' Activation and Time Distributor '''
activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'tanh': nn.Tanh()
}
    

class Activator(nn.Module):
    def __init__(self, activation_name, **kwargs):
        super(Activator, self).__init__(**kwargs)
        self.support_masking = True
        self.act = activations[activation_name]

    def forward(self, x):
        return self.act(x)
    

class TimeDistributor(nn.Module):
    def __init__(self, module, batch_first=False, **kwargs):
        super(TimeDistributor, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        
        x_reshaped = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshaped)

        if self.batch_first:
            # sample, timesteps, outputsize
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            # timesteps, samples, outputsize
            y = y.contiguous().view(-1, x.size(1), y.size(-1))
        return y




