# MSBD5002 Project: Rethinking Symmetric Matrix Factorization for Clustering

## Setup
First, use `env.yaml` to create a new conda environment.

If calling eval_metrics.nmi.nmi() raises an error, please modify `<conda env path>/site-packages\coclust\evaluation\external.py` as follows:

```python
# from sklearn,utils,linear assignment import linear assignment
from scipy.optimize import linear sum assignment as linear assignment
# for row, column in indexes:
for i in range(len(indexes[0])):
    row = indexes[0][i]
    column = indexes[1][i]
    value = cm[row][column]
    total += value
```

## Overall Structure

### `./dataset`

Store the dataset. It should contain the following files:
```
cifar-10-batches-py/
    test_batch
COIL20.mat
reuters21578.mat
TDT2_all.mat
```

### `./implementation`

All our implementation code is stored here. It contains the following subdirectories:

- algorithm/
    - Stores all the optimization algorithms used in the project. 
    - Including Algorithm 1, Algorithm 2 (proposed in the paper)
        and baselines (SymANLS, k-mean, spectral clustering, gradient descent).
    - Please refer to [README for this part](./implementation/algorithm/README.md)
- evaluation/
    - Stores all the evaluation metrics used in the project. 
    - Including AC, NMI.
- loss_function/
    - For experiment part I, we need to implement two different loss functions: with or without regularization term.
    - They are implemented here.
- data_preprocessing/
    - The code used to preprocess and load the dataset.
    - Please refer to [README for this part](./implementation/data_preprocessing/README.md)
- experiment/
    - 1_regularization_term/
        - Code and result for Part I: test the effect of the regularization term.
        - see [README for this part](./implementation/experiment/1_regularization_term/README.md)
    - 2_baselines/
        - Code and result for Part II: baseline algorithm (SymANLS).
        - see [README for this part](./implementation/experiment/2_baselines/README.md)
    - 2_columnwise_fast_update/
        - Code and result for Part II: column-wise fast update (Algo 1 proposed in the paper).
    - 3_baselines/
        - Code and result for Part III: baseline algorithm (k-means and spectral clustering).
    - 3_more_generalized_algorithm/
        - Code and result for Part III: more generalized algorithm (Algo 2 proposed in the paper).

## How to Run the code

For spectural clustering, the code is in a .ipynb file. You can run it in Jupyter Notebook.

For other experiments, there should be a `run_*.sh` script in each experiment directory. You can simply run the script to start the experiment.