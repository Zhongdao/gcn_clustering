###################################################################
# File Name: feeder.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 01:06:16 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import random
import torch
import torch.utils.data as data
class Feeder(data.Dataset):
    '''
    Generate a sub-graph from the feature graph centered at some node, 
    and now the sub-graph has a fixed depth, i.e. 2
    '''
    def __init__(self, feat_path, knn_graph_path, label_path, seed=1, 
                 k_at_hop=[200,5], active_connection=5, train=True):
        np.random.seed(seed)
        random.seed(seed)
        self.features = np.load(feat_path)
        self.knn_graph = np.load(knn_graph_path)[:,:k_at_hop[0]+1]
        self.labels = np.load(label_path)
        self.num_samples = len(self.features)
        self.depth = len(k_at_hop)
        self.k_at_hop = k_at_hop
        self.active_connection = active_connection
        self.train = train
        assert np.mean(k_at_hop)>=active_connection

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix A, together 
        with the indices of the center node and its 1-hop nodes
        '''
        # hops[0] for 1-hop neighbors, hops[1] for 2-hop neighbors
        hops = list()
        center_node = index 
        hops.append(set(self.knn_graph[center_node][1:]))

        # Actually we dont need the loop since the depth is fixed here,
        # But we still remain the code for further revision
        for d in range(1,self.depth): 
            hops.append(set())
            for h in hops[-2]:
                hops[-1].update(set(self.knn_graph[h][1:self.k_at_hop[d]+1]))

        
        hops_set = set([h for hop in hops for h in hop])
        hops_set.update([center_node,])
        unique_nodes_list = list(hops_set) 
        unique_nodes_map = {j:i for i,j in enumerate(unique_nodes_list)}

        center_idx = torch.Tensor([unique_nodes_map[center_node],]).type(torch.long)
        one_hop_idcs = torch.Tensor([unique_nodes_map[i] for i in hops[0]]).type(torch.long)
        center_feat = torch.Tensor(self.features[center_node]).type(torch.float)
        feat = torch.Tensor(self.features[unique_nodes_list]).type(torch.float)
        feat = feat - center_feat
        
        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(unique_nodes_list)
        A = torch.zeros(num_nodes, num_nodes)

        _, fdim = feat.shape
        feat = torch.cat([feat, torch.zeros(max_num_nodes - num_nodes, fdim)], dim=0)
      
        for node in unique_nodes_list:
            neighbors = self.knn_graph[node, 1:self.active_connection+1]
            for n in neighbors:
                if n in unique_nodes_list: 
                    A[unique_nodes_map[node], unique_nodes_map[n]] = 1
                    A[unique_nodes_map[n], unique_nodes_map[node]] = 1

        D = A.sum(1, keepdim=True)
        A = A.div(D)
        A_ = torch.zeros(max_num_nodes,max_num_nodes)
        A_[:num_nodes,:num_nodes] = A

        
        labels = self.labels[np.asarray(unique_nodes_list)]
        labels = torch.from_numpy(labels).type(torch.long)
        #edge_labels = labels.expand(num_nodes,num_nodes).eq(
        #        labels.expand(num_nodes,num_nodes).t())
        one_hop_labels = labels[one_hop_idcs]
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).long()
        
        if self.train:
            return (feat, A_, center_idx, one_hop_idcs), edge_labels, labels

        # Testing
        unique_nodes_list = torch.Tensor(unique_nodes_list)
        unique_nodes_list = torch.cat(
                [unique_nodes_list, torch.zeros(max_num_nodes-num_nodes)], dim=0)
        return(feat, A_, center_idx, one_hop_idcs, unique_nodes_list), edge_labels, labels



