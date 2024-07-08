#!/usr/bin/env python
# coding: utf-8

import torch
from helper.utils import get_hierarchy_relations
import random
import torch.nn as nn
import json
import numpy as np
from helper.utils import get_constr_out

def select_index_multi_dim(feature, index):
    fea = feature[0, index[0], :].unsqueeze(0)
    for i in range(1, index.shape[0]):
        fea = torch.cat((fea, feature[i, index[i], :].unsqueeze(0)),dim=0)
    
    return fea
class ClassificationLoss(torch.nn.Module):
    def __init__(self,
                 taxonomic_hierarchy,
                 label_map,
                 recursive_penalty,
                 recursive_constraint=True):
        """
        Criterion class, classfication loss & recursive regularization
        :param taxonomic_hierarchy:  Str, file path of hierarchy taxonomy
        :param label_map: Dict, label to id
        :param recursive_penalty: Float, lambda value <- config.train.loss.recursive_regularization.penalty
        :param recursive_constraint: Boolean <- config.train.loss.recursive_regularization.flag
        """
        super(ClassificationLoss, self).__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = torch.nn.BCELoss()
        self.recursive_relation = get_hierarchy_relations(taxonomic_hierarchy,
                                                          label_map)
        #0.000001
        self.recursive_penalty = recursive_penalty
        #True
        self.recursive_constraint = recursive_constraint
        self.label_map = label_map
        
        hierarchy_prob_file = './data/rcv1_prob.json'
        f = open(hierarchy_prob_file, 'r', encoding='utf-8')
        hierarchy_prob_str = f.readlines()
        f.close()
        self.hierarchy_prob = json.loads(hierarchy_prob_str[0])
        self.first_level = list(self.hierarchy_prob['Root'].keys())
        second_level = []
        self.second_level_with_child = []
        for key in self.first_level:
            if key in self.hierarchy_prob:
                second_level.extend(list(self.hierarchy_prob[key].keys()))
                for k in self.hierarchy_prob[key].keys():
                    if k in self.hierarchy_prob:
                        self.second_level_with_child.append(k)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos3d = nn.CosineSimilarity(dim=2, eps=1e-6)
    def _recursive_regularization(self, params, device):
        """
        recursive regularization: constraint on the parameters of classifier among parent and children
        :param params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        :param device: torch.device -> config.train.device_setting.device
        :return: loss -> torch.FloatTensor, ()
        """
        rec_reg = 0.0
        for i in range(len(params)):
            if i not in self.recursive_relation.keys():
                continue
            child_list = self.recursive_relation[i]
            if not child_list:
                continue
            child_list = torch.tensor(child_list).to(device)
            child_params = torch.index_select(params, 0, child_list)
            parent_params = torch.index_select(params, 0, torch.tensor(i).to(device))
            parent_params = parent_params.repeat(child_params.shape[0], 1)
            _diff = parent_params - child_params
            diff = _diff.view(_diff.shape[0], -1)
            rec_reg += 1.0 / 2 * torch.norm(diff, p=2) ** 2
        return rec_reg
    
    def _similarity_loss(self, feature, device):
        similairity_loss = torch.tensor(0.0).to(device)
        zero = torch.tensor(0.0).to(device)
        batch = feature.shape[0]
        random_father1 = np.random.choice(list(self.hierarchy_prob['Root'].keys()), batch)
        random_child = [np.random.choice(list(self.hierarchy_prob[random_father1[i]].keys()), 1) for i in range(batch)]
        random_child = torch.tensor([self.label_map[x[0]] for x in random_child]).to(device)
        random_father1 = torch.tensor([self.label_map[x] for x in random_father1]).to(device)
        random_father2 = np.random.choice(list(self.hierarchy_prob['Root'].keys()), batch)
        random_father2 = torch.tensor([self.label_map[x] for x in random_father2]).to(device)
        father1_params = select_index_multi_dim(feature, random_father1)#batch,1,dim
        father2_params = select_index_multi_dim(feature, random_father2)
        child_params = select_index_multi_dim(feature, random_child)
        simf2f = torch.abs(self.cos(father1_params, father2_params))
        simf2c = torch.abs(self.cos(father1_params, child_params))
        _diff = simf2c - simf2f #suppose sim of father-child > sim of father-father
        _diff = torch.sum(torch.exp(-_diff))
        return _diff
    def cal_loss(self, feature, device, level):
        similairity_loss = torch.tensor(0.0).to(device)
        zero = torch.tensor(0.0).to(device)
        batch = feature.shape[0]
        random_father1 = np.random.choice(level, batch)
        random_child = [np.random.choice(list(self.hierarchy_prob[random_father1[i]].keys()), 1) for i in range(batch)]
        random_child = torch.tensor([self.label_map[x[0]] for x in random_child]).to(device)
        random_father1 = torch.tensor([self.label_map[x] for x in random_father1]).to(device)
        random_father2 = np.random.choice(level, batch)
        random_father2 = torch.tensor([self.label_map[x] for x in random_father2]).to(device)
        father1_params = select_index_multi_dim(feature, random_father1)#batch,1,dim
        father2_params = select_index_multi_dim(feature, random_father2)
        child_params = select_index_multi_dim(feature, random_child)
        simf2f = torch.abs(self.cos(father1_params, father2_params))
        simf2c = torch.abs(self.cos(father1_params, child_params))
        _diff = simf2c - simf2f #suppose sim of father-child > sim of father-father
        _diff = torch.sum(torch.exp(-_diff))
        return _diff

    def _similarity_loss1(self, feature, device):
        first_level_loss = self.cal_loss(feature, device, self.first_level)
        second_level_loss = self.cal_loss(feature, device, self.second_level_with_child)
        return first_level_loss + second_level_loss

    def cal_cseloss(self, feature, device, level):
        # batch_size, node_num, node_dim
        similairity_loss = torch.tensor(0.0).to(device)
        zero = torch.tensor(0.0).to(device)
        batch = feature.shape[0]
        random_father1 = np.random.choice(level, batch)
        random_child = [np.random.choice(list(self.hierarchy_prob[random_father1[i]].keys()), 1) for i in range(batch)]
        random_child = torch.tensor([self.label_map[x[0]] for x in random_child]).to(device)
        random_father1 = torch.tensor([self.label_map[x] for x in random_father1]).to(device)
        random_father2 = np.random.choice(level, batch)
        random_father2 = torch.tensor([self.label_map[x] for x in random_father2]).to(device)
        father1_params = select_index_multi_dim(feature, random_father1)#batch,1,dim
        father2_params = select_index_multi_dim(feature, random_father2)
        child_params = select_index_multi_dim(feature, random_child)
        simf2f = self.cos(father1_params, father2_params)
        simf2c = self.cos(father1_params, child_params)
        loss = torch.sum(-torch.log(torch.exp(simf2c/0.05)/(torch.exp(simf2c/0.05)+torch.exp(simf2f/0.05))))
        return loss

    def _cse_loss(self, feature, device):
        first_level_loss = self.cal_cseloss(feature, device, self.first_level)
        second_level_loss = self.cal_cseloss(feature, device, self.second_level_with_child)
        return first_level_loss + second_level_loss
    
    def forward(self, logits, targets, recursive_params, feature, R):
        """
        :param logits: torch.FloatTensor, (batch, N)
        :param targets: torch.FloatTensor, (batch, N)
        :param recursive_params: the parameters on each label -> torch.FloatTensor(N, hidden_dim)
        """
        device = logits.device
        if self.recursive_constraint:
            # org_loss = self.loss_fn(logits, targets)
            logits = torch.sigmoid(logits)
            constr_output = get_constr_out(logits, R)
            train_output = targets * logits.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1 - targets) * constr_output.double() + targets * train_output
            org_loss = self.loss_fn(train_output, targets.double())
            recur_loss = self.recursive_penalty * self._recursive_regularization(recursive_params, device)
            sim_loss = self._cse_loss(feature, device) * 0.0001
            # sim = self._similarity_loss1(feature, device)*0.001
            loss = org_loss
        else:
            org_loss = self.loss_fn(logits, targets)
            sim_loss = self._cse_loss(feature, device)
            recur_loss = 0
            loss = org_loss+recur_loss+sim_loss
        return loss, org_loss, sim_loss, recur_loss
