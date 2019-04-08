# Linkage-based Face Clustering via Graph Convolution Network 
This repository contains the code for our CVPR'19 paper [Linkage-based Face Clustering via GCN](https://arxiv.org/abs/1903.11306), by Zhongdao Wang, Liang Zheng, Yali Li and Shengjin Wang, Tsinghua University and Australian National University.

![](https://github.com/Zhongdao/gcn_clustering/blob/master/asset/pipeline.jpg)

## Introduction
We present an accurate and scalable approach to the face clustering task. We aim at grouping a set of faces by their potential identities. We formulate this task as a link prediction problem: a link exists between two faces if they are of the same identity. The key idea is that
we find the local context in the feature space around an instance(face) contains rich information about the linkage relationship between this instance and its neighbors. By constructing
sub-graphs around each instance as input data,
which depict the local context, we utilize the graph convolution
network (GCN) to perform reasoning and infer the
likelihood of linkage between pairs in the sub-graphs.

## Requirements
- PyTorch 0.4.0
- Python 2.7
- sklearn >= 0.19.1

## Data Format
Firstly, extract features for IJB-B data, and save the features as an NxD dimensional `.npy` file, in which each row is a D-dimensional feature for a sample. Then, save the labels as an Nx1 dimensional `.npy` file, each row is an integer indicating the identity. Lastly, generate the KNN graph (either by brute force or ANN). The KNN graph should be saved as an Nx(K+1) dimensional `.npy` file, and in each row, the first element is the node index, and the following K elements are the indices of its KNN nodes.

For training, featrues+labels+knn_graphs are needed. For testing, only features+knn_graphs are needed, but if you need to compute accuracy the labels are also needed.
We also provide the ArcFace features / labels / knn_graphs of IJB-B/CASIA dataset at [OneDrive](https://1drv.ms/u/s!Ai0390AjdQNVhUbCRARo8PVc1m3j) and [Baidu NetDisk](https://pan.baidu.com/s/1wmMct86Izubw7d2hgBga7A), extract code: 8wj1

## Testing
```
python test.py --val_feat_path path/to/features --val_knn_graph_path path/to/knn/graph --val_labels_path path/to/labels --checkpoint path/to/gcn_weights
```
During inference, the test script will dynamically output the pairwise precision/recall/accuracy. After each subgraph is processed, the test script will output the final B-Cubed precision/recall/F-score (Note that it is not the same as the pairwise p/r/acc) and NMI score.

## Training
```
python train.py --feat_path path/to/features --knn_graph_path path/to/knn/graph --labels_path path/to/labels
```
We employ the CASIA dataset to train the GCN. Usually, 4 epoch is sufficient. We provide a pre-trained model weights in `logs/logs/best.ckpt`

## Citation
If you find GCN-Clustering helps your research, please cite our paper:
```
@inproceedings{wang2019gncclust,
  title={Linkage-based Face Clustering via Graph Convolution Network },
  author={Zhongdao Wang, Liang Zheng, Yali Li and Shengjin Wang},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
