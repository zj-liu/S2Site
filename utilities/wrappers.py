import torch as t
import torch.nn as nn

import numpy as np

from utilities import io_utils

def slice_list_of_arrays(arrays, mask):
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [slice_list_of_arrays(array, mask) for array in arrays]
    else:
        return arrays[mask]


def stack_list_of_arrays(arrays, padded=True):
    if isinstance(arrays[0], tuple) | isinstance(arrays[0], list):
        return [stack_list_of_arrays([array[k] for array in arrays], padded=padded) for k in range(len(arrays[0])) ]
    else:
        if padded:
            return np.concatenate(list(arrays), axis=0)
        else:
            return np.array(list(arrays))


def split_list_of_arrays(arrays, segment_lengths):
    nsplits = len(segment_lengths)
    split_indexes = [0] + list(np.cumsum(segment_lengths))
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [split_list_of_arrays(array, segment_lengths) for array in arrays]
    else:
        return np.array([arrays[split_indexes[i]:split_indexes[i + 1]] for i in range(nsplits)])


def truncate_list_of_arrays(arrays, Ls):
    if isinstance(arrays, tuple) | isinstance(arrays, list):
        return [truncate_list_of_arrays(array, Ls) for array in arrays]
    else:
        if isinstance(Ls, list) | isinstance(Ls, np.ndarray):
            return np.array([array[:L] for array, L in zip(arrays, Ls)])
        else:
            return np.array([array[:Ls] for array in arrays])


class grouped_Predictor_wrapper():
    def __init__(self, multi_inputs=True,multi_outputs=False, verbose=True,
                 input_type = ['frames','points','attributes'], Lmaxs = [800,800,800], Lmax_outputs=None):
        super(grouped_Predictor_wrapper, self).__init__()
        self.multi_inputs = multi_inputs
        self.multi_outputs = multi_outputs
        self.input_type = input_type
        self.Lmax = Lmaxs
        if Lmax_outputs is None:
            self.Lmax_output = Lmaxs[0] if (isinstance(Lmaxs,list) | isinstance(Lmaxs,tuple)) else Lmaxs
        else:
            self.Lmax_output = Lmax_outputs

        self.big_distance = 3e3
        self.big_sequence_distance = 1000
        self.verbose=verbose
        self.wrapper_builder_kwargs = {'multi_inputs':self.multi_inputs,
                                       'multi_outputs':self.multi_outputs,
                                       'input_type':self.input_type,
                                       'Lmaxs':self.Lmax,
                                       'Lmax_outputs':self.Lmax_output,
                                       'verbose':self.verbose
                                       }

    def group_examples(self,Ls):
        Ls = np.array(Ls)
        if isinstance(self.Lmax,list):
            Lmax = self.Lmax[0]
        else:
            Lmax = self.Lmax

        order = np.argsort(Ls)[::-1]
        batches = []
        placed = np.zeros(len(Ls),dtype=np.bool)

        seq_lens = []

        for k in order:
            if not placed[k]:
                if Ls[k]>= Lmax:
                    batches.append( [(k,0,Lmax)] )
                    placed[k] = True
                    seq_lens.append(Lmax)
                else:
                    current_batch = [(k,0, Ls[k] )]
                    placed[k] = True
                    batch_filled = False
                    current_batch_size = Ls[k]

                    while not batch_filled:
                        remaining_size = Lmax - current_batch_size
                        next_example = np.argmax(Ls  - 1e6 * ( placed + (Ls>remaining_size) )    )
                        if (Ls[next_example] <= remaining_size) and not placed[next_example]:
                            current_batch.append( (next_example, current_batch_size,current_batch_size+Ls[next_example] ) )
                            current_batch_size += Ls[next_example]
                            placed[next_example] = True
                        else:
                            batch_filled = True

                    _, _, l = current_batch[-1]
                    seq_lens.append(l)
                    batches.append(current_batch)

        return batches, seq_lens 


    def group_and_padd(self, inputs, groups, which='inputs', weights=None, group_lens=None):
        ngroups = len(groups)
        if which == 'inputs':
            multi_valued = self.multi_inputs
            input_types = self.input_type
        else:
            multi_valued = self.multi_outputs
            input_types = 'outputs'

        if weights is not None:
            if multi_valued:
                Ls = np.array([len(input_) for input_ in inputs[0] ])
            else:
                Ls = np.array([len(input_) for input_ in inputs ])
            weights = weights/( (weights*Ls).mean()/ Ls.mean() )

        if multi_valued:
            ninputs = len(inputs)
            grouped_inputs = []

            grouped_masks = []

            for n in range(ninputs):
                if isinstance(self.Lmax, list):
                    Lmax = self.Lmax[n]
                else:
                    Lmax = self.Lmax
                input_type = input_types[n]
                input_ = inputs[n]
                grouped_input = np.zeros([ngroups, Lmax] + list(input_[0].shape[1:]),dtype=input_[0].dtype)
                if input_type in ['indices','triplets']:
                    grouped_input += -1
                for k,group in enumerate(groups):
                    count = 0
                    start = 0
                    for example,_,_ in group:
                        end = min( start + len(input_[example]), Lmax)
                        if end-start>0:
                            grouped_input[k,start:end] = input_[example][:end-start]
                            if input_type == 'frames':
                                grouped_input[k,start:end,0,:] += count * self.big_distance
                            elif input_type =='points':
                                grouped_input[k,start:end] += count * self.big_distance
                            elif input_type == 'indices':
                                if count>0:
                                    grouped_input[k,start:end] += grouped_input[k,start-1] + self.big_sequence_distance
                            elif input_type == 'triplets':
                                if count>0:
                                    grouped_input[k,start:end] += grouped_input[k,:start].max()+1


                            elif (input_types == 'outputs') & (weights is not None):
                                grouped_input[k,start:end] *= weights[example]
                            start += min( len(input_[example]),Lmax)
                        else:
                            print(n,group,example,'Batch already filled; not enough space for this protein')
                        count +=1
                grouped_inputs.append(grouped_input)

                if input_type in ['indices','triplets']:
                    grouped_masks.append( 1 * np.any((grouped_input != -1), axis=-1, keepdims=True) )
                else:
                    grouped_masks.append( 1 * np.any((grouped_input != 0.0), axis=-1, keepdims=True) )
        else:
            if isinstance(self.Lmax,list):
                Lmax = self.Lmax[0]
            else:
                Lmax = self.Lmax
            input_type = input_types
            grouped_inputs = np.zeros([ngroups,Lmax] + list(inputs[0].shape[1:]), dtype=np.float32)
            for k,group in enumerate(groups):
                count = 0
                for example,start,end in group:
                    grouped_inputs[k,start:end] = inputs[example][:Lmax]
                    if input_type == 'frames':
                        grouped_inputs[k,start:end,0,:] += count * self.big_distance
                    elif input_type == 'points':
                         grouped_inputs[k,start:end] += count * self.big_distance
                    elif input_type == 'indices':
                        if count>0:
                            grouped_inputs[k,start:end] += grouped_inputs[k,start-1] + self.big_sequence_distance
                    elif (input_type == 'outputs') & (weights is not None):
                        grouped_inputs[k,start:end] *= weights[example]
                    count +=1

            if input_type in ['indices','triplets']:
                grouped_masks = 1 * np.any((grouped_inputs != -1), axis=-1, keepdims=True)
            elif input_type == 'outputs':
                if group_lens is not None:
                    grouped_masks = np.zeros((ngroups, Lmax))
                    for b, ind in enumerate(group_lens):
                        grouped_masks[b, :ind] = 1
                else:
                    grouped_masks = None
            else:
                grouped_masks = 1 * np.any((grouped_inputs != 0.0), axis=-1, keepdims=True)            
            
        return grouped_inputs, grouped_masks


    def ungroup_and_unpadd(self, grouped_outputs,groups,which='outputs'):
        if which == 'outputs':
            multi_valued = self.multi_outputs
        else:
            multi_valued = self.multi_inputs
        if multi_valued:
            nexamples = sum([len(group) for group in groups[0]])
            noutputs = len(grouped_outputs)
            outputs = [ np.array([None for _ in range(nexamples)],dtype=np.object) for _ in range(noutputs) ]
        else:
            nexamples = sum([len(group) for group in groups])
            outputs = np.array([None for _ in range(nexamples)],dtype=np.object)
            noutputs = 1
        if multi_valued:
            for n in range(noutputs):
                for k, group in enumerate(groups[n]):
                    for example, start, end in group:
                        outputs[n][example] = grouped_outputs[n][k][start:end]
        else:
            for k,group in enumerate(groups):
                for example,start,end in group:
                    outputs[example] = grouped_outputs[k][start:end]
        return outputs

    def fit(self, inputs, outputs, is_tensor=True, **kwargs):
        if self.multi_inputs:
            Ls = [len(input_) for input_ in inputs[0] ]
            ninputs = len(inputs)
        else:
            Ls = [len(input_) for input_ in inputs]
            ninputs = 1

        if 'sample_weight' in kwargs.keys():
            weights = kwargs.pop('sample_weight')
        else:
            weights = None

        if self.verbose:
            print('Generating groups...')

        groups, group_lens = self.group_examples(Ls)

        if self.verbose:
            print('Grouped %s examples in %s groups'%(len(Ls),len(groups)) )
            print('Grouping and padding...')

        grouped_inputs, masks_inputs = self.group_and_padd(inputs,groups)
        grouped_outputs, masks_outputs = self.group_and_padd(outputs,groups,which='outputs',weights=weights, group_lens=group_lens)

        if is_tensor:
            print('In Tensor format...')
            tensor_inputs = []
            tensor_inp_masks = []
            for i in range(ninputs):
                tensor_inputs.append(t.from_numpy(grouped_inputs[i]))
                tensor_inp_masks.append(t.from_numpy(masks_inputs[i])) 
            tensor_outputs = t.from_numpy(grouped_outputs)
            tensor_out_masks = t.from_numpy(grouped_outputs)

            print('===== Wrapping Completed =====')
            return tensor_inputs, tensor_outputs, [tensor_inp_masks, tensor_out_masks, groups, group_lens]
        print('In array format...')
        print('===== Wrapping Completed =====')
        return grouped_inputs, grouped_outputs, [masks_inputs, masks_outputs, groups, group_lens]
