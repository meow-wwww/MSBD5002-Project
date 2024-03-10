# %%
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from collections import Counter

def read_data(top_topics=30):
    '''
    Read the reuters21578 dataset and return the matrix and labels.
    The returned data has already been preprocessed, so that the dimensionality matches exactly the one used in the paper.
    '''
    tdt2 = loadmat('../../dataset/TDT2_all.mat') 
    # download from: http://www.cad.zju.edu.cn/home/dengcai/Data/TDT2/TDT2_all.mat
    
    # count the occurrences of each class
    all_cnt = 0
    counter_result = Counter(tdt2['gnd'].flatten())
    
    label_to_keep = list(counter_result.keys())[:top_topics]
    doc_idx_to_keep = np.where(np.isin(tdt2['gnd'], label_to_keep))[0]
    
    # accelerate the process
    if (np.diff(doc_idx_to_keep) != 1).sum() == 0 and doc_idx_to_keep.shape[0] == doc_idx_to_keep[-1] + 1:
        new_matrix = tdt2['fea'].toarray()[:doc_idx_to_keep.shape[0]]
        new_labels = tdt2['gnd'][:doc_idx_to_keep.shape[0]]
    else:
        new_matrix = tdt2['fea'].toarray()[doc_idx_to_keep]
        new_labels = tdt2['gnd'][doc_idx_to_keep]
    
    assert new_matrix.shape[0] == new_labels.shape[0] and new_matrix.shape[0] == 9394
    assert new_matrix.shape[1] == 36771
    
    return new_matrix, new_labels


