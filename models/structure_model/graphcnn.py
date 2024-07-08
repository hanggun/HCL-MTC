#!/usr/bin/env python
# coding:utf-8
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_
from sklearn.metrics.pairwise import cosine_similarity
import copy
from torch.nn import functional as F
class HierarchyGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_matrix,
                 out_matrix,
                 in_dim,
                 dropout=0.0,
                 device=torch.device('cpu'),
                 root=None,
                 hierarchical_label_dict=None,
                 label_trees=None):
        """
        Graph Convolutional Network variant for hierarchy structure
        original GCN paper:
                Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
                    arXiv preprint arXiv:1609.02907.
        :param num_nodes: int, N
        :param in_matrix: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_matrix: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param layers: int, the number of layers <- config.structure_encoder.num_layer
        :param time_step: int, the number of time steps <- config.structure_encoder.time_step
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        """
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(
            HierarchyGCNModule(num_nodes,
                               in_matrix, out_matrix,
                               in_dim,
                               dropout,
                               device))

    def forward(self, label):
        return self.model[0](label)


class HierarchyGCNModule(nn.Module):
    def __init__(self,
                 num_nodes,
                 in_adj, out_adj,
                 in_dim, dropout, device, in_arc=True, out_arc=True,
                 self_loop=True):
        """
        module of Hierarchy-GCN
        :param num_nodes: int, N
        :param in_adj: numpy.Array(N, N), input adjacent matrix for child2parent (bottom-up manner)
        :param out_adj: numpy.Array(N, N), output adjacent matrix for parent2child (top-down manner)
        :param in_dim: int, the dimension of each node <- config.structure_encoder.node.dimension
        :param dropout: Float, P value for dropout module <- configure.structure_encoder.node.dropout
        :param prob_train: Boolean, train the probability matrix if True <- config.structure_encoder.prob_train
        :param device: torch.device <- config.train.device_setting.device
        :param in_arc: Boolean, True
        :param out_arc: Boolean, True
        :param self_loop: Boolean, True
        """

        super().__init__()
        self.self_loop = self_loop
        self.out_arc = out_arc
        self.in_arc = in_arc
        self.device = device
        assert in_arc or out_arc
        #  bottom-up child sum
        in_prob = in_adj
        #40*40
        # self.adj_matrix = Parameter(torch.Tensor(in_prob))
        self.adj_matrix = Parameter(torch.Tensor(in_prob)).to(device)
        #40*300
        self.edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        #300*1
        self.gate_weight = Parameter(torch.Tensor(in_dim, 1))
        #40*1
        self.bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        
        self.activation = nn.ReLU()
        #40*40
        self.origin_adj = torch.Tensor(np.where(in_adj <= 0, in_adj, 1.0)).to(device)
        # top-down: parent to child
        #40*40
        self.out_adj_matrix = Parameter(torch.Tensor(out_adj))
        #40*300
        self.out_edge_bias = Parameter(torch.Tensor(num_nodes, in_dim))
        #300*1
        self.out_gate_weight = Parameter(torch.Tensor(in_dim, 1))
        #40*1
        self.out_bias_gate = Parameter(torch.Tensor(num_nodes, 1))
        #300*1
        self.loop_gate = Parameter(torch.Tensor(in_dim, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.zero = torch.tensor([1e-6]).to(self.device)
    def reset_parameters(self):
        """
        initialize parameters
        """
        for param in [self.gate_weight, self.loop_gate, self.out_gate_weight]:
            nn.init.xavier_uniform_(param)
        for param in [self.edge_bias, self.out_edge_bias, self.bias_gate]:
            nn.init.zeros_(param)

    def forward(self, inputs):
        """
        :param inputs: torch.FloatTensor, (batch_size, N, in_dim)
        :return: message_ -> torch.FloatTensor (batch_size, N, in_dim)
        """
        # sim_matrix = F.cosine_similarity(inputs[..., None, :, :], inputs[..., :, None, :], dim=-1)
        # sim_matrix = torch.cdist(inputs, inputs)
        inputs_norm = inputs / inputs.norm(dim=2)[:, :, None]
        sim_matrix = torch.abs(torch.matmul(inputs_norm, inputs_norm.transpose(1,2)))
        #10,40,300输过来的是token output
        h_ = inputs  # batch, N, in_dim
        message_ = torch.zeros_like(h_).to(self.device)  # batch, N, in_dim
        
        h_in_ = torch.matmul(self.origin_adj * self.adj_matrix * sim_matrix, h_)  # batch, N, in_dim
        #加法要与后面两维相等
        in_ = h_in_ + self.edge_bias #batch, N, dim
        in_ = in_
        # batch, N, 1
        in_gate_ = torch.matmul(h_, self.gate_weight)
        # batch, N, 1
        in_gate_ = in_gate_ + self.bias_gate#bias_gate num_nodes, 1
        #batch, N, in_dim @ batch, N, 1 = batch, N, in_dim
        in_ = in_ * F.sigmoid(in_gate_)
        in_ = self.dropout(in_)
        message_ += in_  # batch, N, in_dim
        
        # if torch.isnan(torch.sum(in_)):
        #     raise TypeError("have nan")
        #batch, N, in_dim
        h_output_ = torch.matmul(self.origin_adj.transpose(0, 1) * self.out_adj_matrix * sim_matrix, h_)
        #batch, N, in_dim
        out_ = h_output_ + self.out_edge_bias
        #batch, N, in_dim @ in_dim, 1 = batch, N, 1
        out_gate_ = torch.matmul(h_, self.out_gate_weight)
        #out_bias_gate num_nodes, 1
        out_gate_ = out_gate_ + self.out_bias_gate
        #batch, N, in_dim * N, 1 = batch, N, in_dim
        out_ = out_ * F.sigmoid(out_gate_)
        out_ = self.dropout(out_)
        message_ += out_
        
        # if torch.isnan(torch.sum(out_)):
        #     raise TypeError("have nan")
            
        #batch, N, in_dim @ in_dim, 1 = batch, N, 1
        loop_gate = torch.matmul(h_, self.loop_gate)
        #batch, N, in_dim * batch, N, 1 = batch, N, in_dim
        loop_ = h_ * F.sigmoid(loop_gate)
        loop_ = self.dropout(loop_)
        message_ += loop_
        
        #message = in_ + out_ + loop_
        
        return self.activation(message_)
