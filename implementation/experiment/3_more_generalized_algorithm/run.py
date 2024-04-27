# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# read data
import sys
sys.path.append('../..')
import data_preprocessing
from importlib import reload
import importlib

# training process
import algorithm.algo2 as algo2

# evaluation
import algorithm.kmeans as kmeans
import eval_metrics.ac
import eval_metrics.nmi

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[Using device]', device)

# %%
import argparse

parser = argparse.ArgumentParser(description='Train a model for clustering')
parser.add_argument('--dataset', type=str, help='dataset name')
parser.add_argument('--n_clusters', type=int, help='number of clusters')
parser.add_argument('--classes', type=str, default=None, help='classes to use')
parser.add_argument('--tol', type=float, default=1e-6, help='tolerance')
parser.add_argument('--filename', type=str, help='filename to save the result')

args = parser.parse_args()

print(f'[args] {args}')

f = open(args.filename, 'a')

# %%
# DATASET = getattr(data_preprocessing, args.dataset)
DATASET = importlib.import_module(f'data_preprocessing.{args.dataset}')
CLUSTER_NUM = args.n_clusters
CLASS_LIST = eval(args.classes) if args.classes is not None else None
if CLASS_LIST is not None:
    CLASS_LIST.sort()
    assert len(CLASS_LIST) == CLUSTER_NUM, 'The length of class list should be equal to n_clusters'
TOL = args.tol

# %%
# read data
mat, label = DATASET.read_data(num_of_class=CLUSTER_NUM, random_shuffle=True, 
                              random_choose_class=CLASS_LIST)
mat = torch.tensor(mat, dtype=torch.float32)
mat -= torch.mean(mat)

print(f'[Using classes] {np.unique(np.array(label))}')

# compute the similarity matrix using cosine similarity
mat_normalized = mat / torch.norm(mat, dim=1).reshape(-1, 1)
A = mat_normalized @ mat_normalized.T
A.to(device)
pass

# %%
H = torch.randn(A.shape[0], CLUSTER_NUM)
H.to(device)

H_out = algo2.matrixFac(A=A, H=H, 
                        tol=1e-6,
                        max_iter=100000000,
                        print_every=100)

# %%
pred_labels, centroids = kmeans.kmeans_optimal(H_out, CLUSTER_NUM, turns=40)
acc = eval_metrics.ac.ac(label, pred_labels)
nmi = eval_metrics.nmi.nmi(label, pred_labels)
print(f'[Accuracy] {acc}\n[NMI] {nmi}')
print()


print(f"{args.dataset},{CLUSTER_NUM},{'/'.join(list(map(str, np.unique(np.array(label)).tolist())))},{TOL},{acc},{nmi}", file=f)

f.close()

# %%



