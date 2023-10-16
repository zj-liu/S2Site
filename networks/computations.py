import numpy as np

import torch
import torch.nn as nn

from networks.other_components import *


class Frame_Builder(nn.Module):
    def __init__(self, order='1', dipole=False, mask=False, device='cpu'):
        super(Frame_Builder, self).__init__()
        # order: The permutation to build the frame (for order ='2', the zaxis is defined from X_1 -> X_3).
        # dipole: Only used if using a quadruplet of indices. Then, frame has a 4rth additional direction (the dipole).
        self.mask = mask
        self.reset_device(device)

        self.epsilon = 1e-6
        self.order = order
        self.dipole = dipole
        self.triplet_shape = None
        self.point_shape = None

    def forward(self, frame_indices, point_clouds, mask):
        '''
        For each atom, four cases should be distinguished.
        Case 1: both neighbors exist (i.e. at least two covalent bonds). Construct the frame as usual using Schmidt orthonormalization.
        Case 2: The first neighbor does not exist (i.e. the next atom along the protein tree. Example: Alanine, A_i =Cbeta A_{i-1} = Calpha, A_{i+1} = Cgamma does not exists). 
        The solution is to place a virtual atom such that (A_{i-2}, A_{i-1}, A_{i}, A_{i+1,virtual}) is a parallelogram. For alanine, (N,Calpha,Cbeta,Cgamma_virt) is a parallelogram.
        Case 3: The second neighbor does not exist (i.e. the previous atom along the protein tree.
        Example: N-terminal N along the backbone, or missing residues). Similarly, we build a parallelogram
        (A_{i-1,virtual}, A_i, A_{i+1},A_{i+2}).
        Case 4: None exist (missing atoms). Use the default cartesian frame.
        '''

        triplets = torch.clamp(frame_indices, 0, point_clouds.size(-2)-1)
        self.triplet_shape = triplets.shape
        self.point_shape = point_clouds.shape

        delta_10 = slice_computation(points=point_clouds, triplets=triplets, axis_i=1, axis_j=0)
        delta_20 = slice_computation(points=point_clouds, triplets=triplets, axis_i=2, axis_j=0)

        if self.order in ['2', '3']: 
            # Order 1: the second point is on the axis-z and the third in the xz plane. 
            # Order 2: the third point is on the axis-z and the second in the xz plane.
            delta_10, delta_20 = delta_20, delta_10

        centers = slice_computation(points=point_clouds, triplets=triplets, axis_i=0)

        axis_z = (delta_10 + self.epsilon * torch.unsqueeze(self.z, 0)) / ( torch.sqrt( torch.sum(delta_10**2, dim=-1, keepdim=True) ) + self.epsilon)

        axis_y = torch.cross(axis_z, delta_20)
        axis_y = (axis_y + self.epsilon * torch.unsqueeze(self.y, 0)) / ( torch.sqrt( torch.sum(axis_y**2, dim=-1, keepdim=True) ) + self.epsilon)

        axis_x = torch.cross(axis_y, axis_z)
        axis_x = (axis_x + self.epsilon * torch.unsqueeze(self.x, 0)) / ( torch.sqrt( torch.sum(axis_x**2, dim=-1, keepdim=True) ) + self.epsilon)

        if self.order == '3':
            axis_x, axis_y, axis_z = axis_z, axis_x, axis_y
        
        if self.dipole:
            dipole = slice_computation(points=point_clouds, triplets=triplets, axis_i=3, axis_j=0)
            dipole = ( dipole + self.epsilon * torch.unsqueeze(self.z, 0) ) / ( torch.sqrt( torch.sum(dipole**2, dim=-1, keepdim=True) ) + self.epsilon)
            frames = torch.stack([centers, axis_x, axis_y, axis_z, dipole], dim=-2)
        else:
            frames = torch.stack([centers, axis_x, axis_y, axis_z], dim=-2)

        # B, L, 4, 3
        return frames, self.compute_mask(mask)


    def compute_mask(self, mask):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if self.dipole:
            num_vectors = 5
        else:
            num_vectors = 4

        if mask not in [None, [None, None]]:
            mask = torch.unsqueeze(mask[0], dim=-1)
            return mask.repeat([1, 1, num_vectors] + list(mask.shape[3:]))   
        return mask


    def reset_device(self, device):
        self.x = torch.tensor([[1, 0, 0]], dtype=torch.float).to(device)
        self.y = torch.tensor([[0, 1, 0]], dtype=torch.float).to(device)
        self.z = torch.tensor([[0, 0, 1]], dtype=torch.float).to(device)


class Neighborhood_Computation(nn.Module):
    def __init__(self, K_neighbor, coordinates=['euclidian'], 
                    self_neighborhood=True, index_distance_max=None, nrotations=1, input_len=2, mask=False, device='cpu'):
        super(Neighborhood_Computation, self).__init__()
        
        self.Kmax = K_neighbor
        self.coordinates = coordinates
        self.self_neighborhood = self_neighborhood
        self.mask = mask
        
        for coordinate in self.coordinates:
            assert coordinate in ['distance','index_distance',
                                  'euclidian','ZdotZ','ZdotDelta','dipole_spherical']

        self.first_format = []
        self.second_format = []
        if ('euclidian' in self.coordinates) | ('ZdotZ' in self.coordinates) | ('ZdotDelta' in self.coordinates):
            self.first_format.append('frame')
            if self.self_neighborhood | ('ZdotZ' in self.coordinates) | ('ZdotDelta' in self.coordinates):
                self.second_format.append('frame')
            else:
                self.second_format.append('point')
        elif 'distance' in self.coordinates:
            self.first_format.append('point')
            self.second_format.append('point')
        
        
        if 'index_distance' in self.coordinates:
            self.first_format.append('index')
            self.second_format.append('index')

        coordinates_dimension = 0
        for coordinate in coordinates:
            if coordinate == 'euclidian':
                coordinates_dimension += 3
            elif coordinate =='dipole_spherical':
                coordinates_dimension += 2
            elif coordinate == 'ZdotDelta':
                coordinates_dimension += 2
            else:
                coordinates_dimension += 1

        self.coordinates_dimension = coordinates_dimension

        self.index_distance_max = index_distance_max

        self.epsilon = 1e-10
        self.big_distance = 1000.0
        self.nrotations = nrotations
        if self.nrotations > 1:
            assert self.coordinates == ['euclidian'], 'Rotations only work with Euclidian coordinates'

        
        # build part
        self.input_len = input_len
        self.nattributes = input_len - len(self.first_format) - (1 - 1 * self.self_neighborhood) * len(self.second_format)

        self.reset_device(device)


    def forward(self, inputs, mask):
        # inputs = frame_feats, attr

        if mask is None:
            mask = [None for _ in inputs]

        if 'frame' in self.first_format:
            first_frame = inputs[self.first_format.index('frame')]
        else:
            first_frame = None
        if 'frame' in self.second_format:
            if self.self_neighborhood:
                second_frame = first_frame
            else:
                second_frame = inputs[len(self.first_format) + self.second_format.index('frame')]
        else:
            second_frame = None

        if 'point' in self.first_format:
            first_point = inputs[self.first_format.index('point')]
        else:
            first_point = None

        if 'point' in self.second_format:
            if self.self_neighborhood:
                second_point = first_point
            else:
                second_point = inputs[len(self.first_format) + self.second_format.index('point')]
        else:
            second_point = None

        if 'index' in self.first_format:
            first_index = inputs[self.first_format.index('index')]
        else:
            first_index = None
        if 'index' in self.second_format:
            if self.self_neighborhood:
                second_index = first_index
            else:
                second_index = inputs[len(self.first_format) + self.second_format.index('index')]
        else:
            second_index = None

        second_attributes = inputs[-self.nattributes:]

        first_mask = mask[0]
        if (first_mask is not None)  and (first_frame is not None):
            first_mask = first_mask[:,:,1]

        if self.self_neighborhood:
            second_mask = first_mask
        else:
            second_mask = mask[len(self.first_format)]
            if (second_mask is not None) and (second_frame is not None):
                second_mask = second_mask[:, :, 1]

        if second_mask is not None:
            irrelevant_seconds = torch.unsqueeze(torch.squeeze(1 - second_mask.float(), dim=-1), dim=1).float()
        else:
            irrelevant_seconds = None
        
        if first_frame is not None:
            first_center = first_frame[:,:,0]
            ndims = 3
        elif first_point is not None:
            first_center = first_point
            ndims = 3
        else:
            first_center = first_index.float()
            ndims = 1

        if second_frame is not None:
            second_center = second_frame[:,:,0]
        elif second_point is not None:
            second_center = second_point
        else:
            second_center = second_index.float()
        
        # B, L, L == 108, 9216, 9216
        distance_square = distance(first_center, second_center, squared=True, ndims=ndims)
        if irrelevant_seconds is not None:
            distance_square += irrelevant_seconds * self.big_distance

        if irrelevant_seconds is not None:
            distance_square += irrelevant_seconds * self.big_distance

        # B, L, K == 108, 9216, 16
        _, neighbors = torch.sort(distance_square, stable=True)
        neighbors = neighbors[:, :, :self.Kmax]
        #neighbors = torch.unsqueeze(neighbors[:, :, :self.Kmax], dim=-1)
        
        # B, L, K, 12
        neighbors_attributes = [get_values_from_indices(attributes, neighbors) 
                                for attributes in second_attributes]

        neighbor_coordinates = []

        if 'euclidian' in self.coordinates:
            euclidian_coordinates = torch.sum(torch.unsqueeze(
                # B x Lmax x Kmax x 3
                get_values_from_indices(second_center, neighbors) 
                - torch.unsqueeze(first_center, dim=-2),  # B X Lmax X 1 X 3,
                dim=-2)  # B X Lmax X Kmax X 1 X 3 \
                * torch.unsqueeze(
                first_frame[:,:,1:4],
                dim=-3)  # B X Lmax X 1 X 3 X 3
                , dim=-1)  # B X Lmax X Kmax X 3

            # Note - check effect of rotation in dim=-2 if neccessary
            if self.nrotations>1:
                euclidian_coordinates = torch.dot(
                euclidian_coordinates, self.rotations)

                neighbors_attributes = [torch.unsqueeze(
                    neighbors_attribute, dim=-2) for neighbors_attribute in neighbors_attributes]

            neighbor_coordinates.append(euclidian_coordinates)
        
        if 'dipole_spherical' in self.coordinates:
                dipole_euclidian_coordinates = torch.sum(torch.unsqueeze(
                    # B x Lmax x Kmax x 3
                    get_values_from_indices(second_frame[:,:,-1], neighbors),
                    dim=-2)  # B X Lmax X Kmax X 1 X 3 \
                    * torch.unsqueeze(
                    first_frame[:,:,1:4],
                    dim=-3)  # B X Lmax X 1 X 3 X 3
                    , dim=-1)  # B X Lmax X Kmax X 3
                dipole_spherical_coordinates = euclidian_to_spherical(dipole_euclidian_coordinates, return_r=False)
                neighbor_coordinates.append(dipole_spherical_coordinates)

        if 'distance' in self.coordinates:
            distance_neighbors = torch.unsqueeze(torch.sqrt(get_values_from_indices_2d(
                distance_square, neighbors) ), axis=-1)

            neighbor_coordinates.append(distance_neighbors)

        if 'ZdotZ' in self.coordinates:
            first_zdirection = first_frame[:,:,-1]
            second_zdirection = second_frame[:, :, -1]

            ZdotZ_neighbors = torch.sum(torch.unsqueeze(
                first_zdirection, dim=-2) * get_values_from_indices(second_zdirection, neighbors), dim=-1, keepdims=True)
            neighbor_coordinates.append(ZdotZ_neighbors)

        if 'ZdotDelta' in self.coordinates:
            first_zdirection = first_frame[:,:,-1]
            second_zdirection = second_frame[:, :, -1]

            DeltaCenter_neighbors = (get_values_from_indices(
                second_center, neighbors) - torch.unsqueeze(first_center, dim=-2)) / (distance_neighbors + self.epsilon)
            ZdotDelta_neighbors = torch.sum(torch.unsqueeze(
                first_zdirection, dim=-2) * DeltaCenter_neighbors, dim=-1, keepdims=True)
            DeltadotZ_neighbors = torch.sum(DeltaCenter_neighbors * get_values_from_indices(
                second_zdirection, neighbors), dim=-1, keepdims=True)

            neighbor_coordinates.append(DeltadotZ_neighbors)
            neighbor_coordinates.append(ZdotDelta_neighbors)

        if 'index_distance' in self.coordinates:
            index_distance = torch.abs( (
                torch.unsqueeze(first_index, dim=-2) - get_values_from_indices(second_index, neighbors) ).float() )

            if self.index_distance_max is not None:
                index_distance = torch.clamp(index_distance, 0, self.index_distance_max)

            neighbor_coordinates.append(index_distance)


        neighbor_coordinates = torch.cat(neighbor_coordinates, -1)

        if first_mask is not None:
            if (self.nrotations > 1):
                neighbor_coordinates *= torch.unsqueeze(torch.unsqueeze(
                    first_mask.float(), dim=-1),dim=-1)

                for neighbors_attribute in neighbors_attributes:
                    neighbors_attribute *= torch.unsqueeze(torch.unsqueeze(
                        first_mask.float(), dim=-1),dim=-1)
            else:
                neighbor_coordinates = neighbor_coordinates * torch.unsqueeze(first_mask.float(), dim=-1)

                for neighbors_attribute in neighbors_attributes:
                    neighbors_attribute = neighbors_attribute * torch.unsqueeze(first_mask.float(), dim=-1)

        output = [neighbor_coordinates] + neighbors_attributes
        
        # [(B, L=9216, K=16, 3), (B, L=9216, K=16, 12)]
        # [(B, L=1024, 1), (B, L=1024, K2=14, feat_dim=64), (B, L=1024, K2=14, feat_dim=64)]
        return output, self.compute_mask(inputs, mask)

    def compute_mask(self, inputs, mask):
        # Just pass the received mask from previous layer, to the next layer or
        # manipulate it if this layer changes the shape of the input
        if mask not in [None, [None for _ in inputs]]:
            first_mask = mask[0]
            if 'frame' in self.first_format:
                first_mask = first_mask[:, :, 1]

            if self.nrotations>1:
                return [torch.unsqueeze(torch.unsqueeze(first_mask,axis=-1), dim=-1) ]* (1+ self.nattributes)
            else:
                return [torch.unsqueeze(first_mask,dim=-1) ] * (1+self.nattributes)
        else:
            return mask

    def reset_device(self, device):
        if self.nrotations > 1:
            phis = torch.arange(self.nrotations) / self.nrotations * 2 * torch.pi
            rotations = torch.zeros([self.nrotations, 3, 3], dtype=torch.float32).to(device)
            rotations[:, 0, 0] = torch.cos(phis)
            rotations[:, 1, 1] = torch.cos(phis)
            rotations[:, 1, 0] = torch.sin(phis)
            rotations[:, 0, 1] = -torch.sin(phis)
            rotations[:, 2, 2] = 1
            self.rotations = rotations
