import torch
import torch.nn as nn


class Attention_Layer(nn.Module):
    def __init__(self, self_attention=True, beta=True):
        super(Attention_Layer, self).__init__()
        self.self_attention = self_attention
        self.beta = beta
        self.epsilon = 1e-6

        self.first_ = True

        self.Lmax = None
        self.Kmax = None
        self.nfeatures_graph = None
        self.nheads = None
        self.nfeatures_output = None

    def forward(self, inputs, mask):
        if self.beta and self.self_attention:
            beta, self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        elif self.beta and (~self.self_attention):
            beta, attention_coefficients, node_outputs, graph_weights = inputs
        elif (~self.beta) and self.self_attention:
            self_attention, attention_coefficients, node_outputs, graph_weights = inputs
        else:
            attention_coefficients, node_outputs, graph_weights = inputs

        if self.first_:
            self.Lmax = graph_weights.shape[1]
            self.Kmax = graph_weights.shape[2]
            self.nfeatures_graph = graph_weights.shape[-1]
            self.nheads = attention_coefficients.shape[-1] // self.nfeatures_graph
            self.nfeatures_output = node_outputs.shape[-1] // self.nheads
            self.first_ = False

        if self.beta:
            beta = torch.reshape(
                beta, [-1, self.Lmax, self.nfeatures_graph, self.nheads])

        if self.self_attention:
            self_attention = torch.reshape(
                self_attention, [-1, self.Lmax, self.nfeatures_graph, self.nheads])

        attention_coefficients = torch.reshape(
            attention_coefficients, [-1, self.Lmax, self.Kmax, self.nfeatures_graph, self.nheads])

        node_outputs = torch.reshape(
            node_outputs, [-1, self.Lmax, self.Kmax, self.nfeatures_output, self.nheads])

        # Add self-attention coefficient.
        if self.self_attention:
            attention_coefficients_self, attention_coefficient_others = torch.split(
                attention_coefficients, [1, self.Kmax - 1], dim=2)
            attention_coefficients_self = attention_coefficients_self + torch.unsqueeze(self_attention, dim=2)
            attention_coefficients = torch.cat(
                [attention_coefficients_self, attention_coefficient_others], dim=2)


        # Multiply by inverse temperature beta.
        if self.beta:
            attention_coefficients = attention_coefficients * torch.unsqueeze(beta + self.epsilon, dim=2)

        tmp_coefficients, _ = torch.max(attention_coefficients, dim=-3, keepdim=True)
        tmp_coefficients, _ = torch.max(tmp_coefficients, dim=-2, keepdim=True)
        attention_coefficients -= tmp_coefficients

        attention_coefficients_final = torch.sum(torch.unsqueeze(
            graph_weights, dim=-1) * torch.exp(attention_coefficients), dim=-2)
        attention_coefficients_final = attention_coefficients_final / (torch.sum(
            torch.abs(attention_coefficients_final), dim=-2, keepdim=True) + self.epsilon)

        output_final = torch.reshape(torch.sum(node_outputs * torch.unsqueeze(
            attention_coefficients_final, dim=-2), dim=2), [-1, self.Lmax, self.nfeatures_output * self.nheads])

        return [output_final, attention_coefficients_final], self.compute_mask(inputs, mask)
    

    def compute_mask(self, input, mask):
        if mask not in [None,[None for _ in input]]:
            if self.beta | self.self_attention:
                return [mask[0], None]
            else:
                return [None,None]
        else:
            return mask

