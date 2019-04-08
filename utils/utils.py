###################################################################
# File Name: utils.py
# Author: Zhongdao Wang
# mail: wcd17@mails.tsinghua.edu.cn
# Created Time: Tue 28 Aug 2018 04:57:29 PM CST
###################################################################

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
from scipy.sparse import coo_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def norm(X):
    for ix,x in enumerate(X):
        X[ix]/=np.linalg.norm(x)
    return X

def plot_embedding(X,Y):
    x_min, x_max = np.min(X,0), np.max(X,0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10,10))
    for i in xrange(X.shape[0]):
        plt.text(X[i,0],X[i,1], str(Y[i]),
                color=plt.cm.Set1(Y[i]/10.),
                fontdict={'weight':'bold','size':12})
    plt.savefig('a.jpg')


EPS = np.finfo(float).eps


def contingency_matrix(ref_labels, sys_labels):
    """Return contingency matrix between ``ref_labels`` and ``sys_labels``."""
    ref_classes, ref_class_inds = np.unique(ref_labels, return_inverse=True)
    sys_classes, sys_class_inds = np.unique(sys_labels, return_inverse=True)
    n_frames = ref_labels.size
    # Following works because coo_matrix sums duplicate entries. Is roughly
    # twice as fast as np.histogram2d.
    cmatrix = coo_matrix(
        (np.ones(n_frames), (ref_class_inds, sys_class_inds)),
        shape=(ref_classes.size, sys_classes.size),
        dtype=np.int)
    cmatrix = cmatrix.toarray()
    return cmatrix, ref_classes, sys_classes


def bcubed(ref_labels, sys_labels, cm=None):
    """Return B-cubed precision, recall, and F1.

    The B-cubed precision of an item is the proportion of items with its
    system label that share its reference label (Bagga and Baldwin, 1998).
    Similarly, the B-cubed recall of an item is the proportion of items
    with its reference label that share its system label. The overall B-cubed
    precision and recall, then, are the means of the precision and recall for
    each item.

    Parameters
    ----------
    ref_labels : ndarray, (n_frames,)
        Reference labels.

    sys_labels : ndarray, (n_frames,)
        System labels.

    cm : ndarray, (n_ref_classes, n_sys_classes)
        Contingency matrix between reference and system labelings. If None,
        will be computed automatically from ``ref_labels`` and ``sys_labels``.
        Otherwise, the given value will be used and ``ref_labels`` and
        ``sys_labels`` ignored.
        (Default: None)

    Returns
    -------
    precision : float
        B-cubed precision.

    recall : float
        B-cubed recall.

    f1 : float
        B-cubed F1.

    References
    ----------
    Bagga, A. and Baldwin, B. (1998). "Algorithms for scoring coreference
    chains." Proceedings of LREC 1998.
    """
    if cm is None:
        cm, _, _ = contingency_matrix(ref_labels, sys_labels)
    cm = cm.astype('float64')
    cm_norm = cm / cm.sum()
    precision = np.sum(cm_norm * (cm / cm.sum(axis=0)))
    recall = np.sum(cm_norm * (cm / np.expand_dims(cm.sum(axis=1), 1)))
    f1 = 2*(precision*recall)/(precision + recall)
    return precision, recall, f1
