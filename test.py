###################################################################
# File Name: train.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Thu 06 Sep 2018 10:08:49 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import os.path as osp
import sys
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader

import model
from feeder.feeder import Feeder
from utils import to_numpy
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint 
from utils.utils import bcubed
from utils.graph import graph_propagation, graph_propagation_soft, graph_propagation_naive

from sklearn.metrics import normalized_mutual_info_score, precision_score, recall_score

def single_remove(Y, pred):
    single_idcs = np.zeros_like(pred)
    pred_unique = np.unique(pred)
    for u in pred_unique:
        idcs = pred == u
        if np.sum(idcs) == 1:
            single_idcs[np.where(idcs)[0][0]] = 1
    remain_idcs = [i for i in range(len(pred)) if not single_idcs[i]]
    remain_idcs = np.asarray(remain_idcs)
    return Y[remain_idcs], pred[remain_idcs]

def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True
    
    valset = Feeder(args.val_feat_path,
                    args.val_knn_graph_path,
                    args.val_label_path,
                    args.seed,
                    args.k_at_hop,
                    args.active_connection, 
                    train=False)
    valloader = DataLoader(
            valset, batch_size=args.batch_size,
            num_workers=args.workers, shuffle=False, pin_memory=True)

    ckpt = load_checkpoint(args.checkpoint)
    net = model.gcn()
    net.load_state_dict(ckpt['state_dict'])
    net = net.cuda()

    knn_graph = valset.knn_graph
    knn_graph_dict = list() 
    for neighbors in knn_graph:
        knn_graph_dict.append(dict())
        for n in neighbors[1:]:
            knn_graph_dict[-1][n] = []

    criterion = nn.CrossEntropyLoss().cuda()
    edges, scores = validate(valloader, net, criterion)
    
    np.save('edges', edges)
    np.save('scores', scores)
    #edges=np.load('edges.npy')
    #scores = np.load('scores.npy')
    
    clusters = graph_propagation(edges, scores, max_sz=900, step=0.6, pool='avg' )
    final_pred = clusters2labels(clusters, len(valset))
    labels = valset.labels

    print('------------------------------------')
    print('Number of nodes: ', len(labels))
    print('Precision   Recall   F-Sore   NMI')
    p,r,f = bcubed(final_pred, labels)
    nmi = normalized_mutual_info_score(final_pred, labels)
    print(('{:.4f}    '*4).format(p,r,f, nmi))

    labels, final_pred = single_remove(labels, final_pred)
    print('------------------------------------')
    print('After removing singleton culsters, number of nodes: ', len(labels))
    print('Precision   Recall   F-Sore   NMI')
    p,r,f = bcubed(final_pred, labels)
    nmi = normalized_mutual_info_score(final_pred, labels)
    print(('{:.4f}    '*4).format(p,r,f, nmi))
    
    
def clusters2labels(clusters, n_nodes):
    labels = (-1)* np.ones((n_nodes,))
    for ci, c in enumerate(clusters):
        for xid in c:
            labels[xid.name] = ci
    assert np.sum(labels<0) < 1
    return labels 

def make_labels(gtmat):
    return gtmat.view(-1)

def validate(loader, net, crit):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs  = AverageMeter()
    precisions  = AverageMeter()
    recalls  = AverageMeter()
    
    net.eval()
    end = time.time()
    edges = list()
    scores = list()
    for i, ((feat, adj, cid, h1id, node_list), gtmat) in enumerate(loader):
        data_time.update(time.time() - end)
        feat, adj, cid, h1id, gtmat = map(lambda x: x.cuda(), 
                                (feat, adj, cid, h1id, gtmat))
        pred = net(feat, adj, h1id)
        labels = make_labels(gtmat).long()
        loss = crit(pred, labels)
        pred = F.softmax(pred, dim=1)
        p,r, acc = accuracy(pred, labels)
        
        losses.update(loss.item(),feat.size(0))
        accs.update(acc.item(),feat.size(0))
        precisions.update(p, feat.size(0))
        recalls.update(r,feat.size(0))
    
        batch_time.update(time.time()- end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\n'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})\t'
                  'Precison {precisions.val:.3f} ({precisions.avg:.3f})\t'
                  'Recall {recalls.val:.3f} ({recalls.avg:.3f})'.format(
                        i, len(loader), batch_time=batch_time,
                        data_time=data_time, losses=losses, accs=accs, 
                        precisions=precisions, recalls=recalls))

	node_list = node_list.long().squeeze().numpy()
        bs = feat.size(0)
        for b in range(bs): 
            cidb = cid[b].int().item() 
            nl = node_list[b]

            for j,n in enumerate(h1id[b]):
                n = n.item()
	        edges.append([nl[cidb], nl[n]])
                scores.append(pred[b*args.k_at_hop[0]+j,1].item())
                #print(nl[cidb],nl[n])
                #print(pred[b*bs+j,1].item(),labels[b*bs+j])
            #time.sleep(10)
    edges = np.asarray(edges)
    scores = np.asarray(scores)
    return edges, scores

def accuracy(pred, label):
    pred = torch.argmax(pred, dim=1).long()
    acc = torch.mean((pred == label).float())
    pred = to_numpy(pred)
    label = to_numpy(label)
    p = precision_score(label, pred)
    r = recall_score(label, pred)
    return p,r,acc 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # misc
    working_dir = osp.dirname(osp.abspath(__file__)) 
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--print_freq', default=40, type=int)

    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--k-at-hop', type=int, nargs='+', default=[20,5])
    parser.add_argument('--active_connection', type=int, default=5)

    # Validation args 
    parser.add_argument('--val_feat_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/1024.fea.npy'))
    parser.add_argument('--val_knn_graph_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/knn.graph.1024.bf.npy'))
    parser.add_argument('--val_label_path', type=str, metavar='PATH',
                        default=osp.join(working_dir, '../facedata/1024.labels.npy'))

    # Test args
    parser.add_argument('--checkpoint', type=str, metavar='PATH')
    args = parser.parse_args()
    main(args)
