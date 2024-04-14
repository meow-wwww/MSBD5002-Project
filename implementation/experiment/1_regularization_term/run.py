# %%
import torch
import numpy as np
import matplotlib.pyplot as plt

# read data
import sys
sys.path.append('../..')
import data_preprocessing.coil20 as coil20
from importlib import reload
reload(coil20)

# training process
import algorithm.GradientDescent as GradientDescent
reload(GradientDescent)
from loss_function import origin

# evaluation
import algorithm.kmeans as kmeans
import eval_metrics.ac

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('[Using device]', device)

# %%
import argparse

parser = argparse.ArgumentParser(description='Train a model for clustering')
parser.add_argument('--n_clusters', type=int, default=20, help='number of clusters')
parser.add_argument('--loss', type=str, default='reconstruction_loss_without_regularization', help='loss function')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--lamda', type=float, default=1, help='regularization parameter')
parser.add_argument('--filename', type=str, help='filename to save the result')

args = parser.parse_args()

print(f'[args] {args}')

f = open(args.filename, 'a')

# %%
CLUSTER_NUM = args.n_clusters
LOSS_FUNCTION_NAME = args.loss
LOSS_FUNCTION = getattr(origin, LOSS_FUNCTION_NAME)
LEARNING_RATE = args.lr
if LOSS_FUNCTION_NAME == 'reconstruction_loss_without_regularization':
    LAMBDA = None
else:
    LAMBDA = args.lamda
    


# %%
# read data
mat, label = coil20.read_data(num_of_class=CLUSTER_NUM, random_shuffle=True, 
                              random_choose_class=True)
mat = torch.tensor(mat, dtype=torch.float32)
mat -= torch.mean(mat)

print(f'[Using classes] {np.unique(np.array(label))}')

# compute the similarity matrix using cosine similarity
mat_normalized = mat / torch.norm(mat, dim=1).reshape(-1, 1)
A = mat_normalized @ mat_normalized.T
A.to(device)
pass

# %%
N = A.shape[0]
H = torch.randn(N, CLUSTER_NUM)
H.to(device)
H.requires_grad = True
learning_rate = LEARNING_RATE

H_out, final_loss = GradientDescent.GradientDescent(A=A, H=H, lamda=LAMBDA, 
                                        loss_fn=LOSS_FUNCTION, 
                                        lr=learning_rate, max_iter=1000000,
                                        tol=learning_rate*1e-4, print_every=1000)

print(f'final loss without regularization: {final_loss}')

# %%
H_out = H_out.detach().cpu().numpy()
pred_labels, centroids = kmeans.kmeans(H_out, CLUSTER_NUM)
acc = eval_metrics.ac.ac(label, pred_labels)
print(f'[Accuracy] {acc}')


print(f"{CLUSTER_NUM},{'/'.join(list(map(str, np.unique(np.array(label)).tolist())))},{LOSS_FUNCTION.__name__},{LEARNING_RATE},{LAMBDA},{final_loss},{acc}", file=f)

f.close()

# %%



