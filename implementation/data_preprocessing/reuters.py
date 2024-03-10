# %%
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np

# %%
def read_reuters21578():
    '''
    Read the reuters21578 dataset and return the matrix and labels.
    The returned data has already been preprocessed, so that the dimensionality matches exactly the one used in the paper.
    '''
    reuters21578 = loadmat('../../dataset/reuters21578.mat') 
    # download from: http://www.cad.zju.edu.cn/home/dengcai/Data/Reuters21578/Reuters21578.mat
    
    # count the occurrences of each class
    topic_count = {}
    for topic in reuters21578['gnd']:
        topic = topic[0]
        if topic not in topic_count:
            topic_count[topic] = 1
        else:
            topic_count[topic] += 1

    # check documents with topics that occur less than 10 times
    to_be_removed_doc_idx = []
    for doc_idx in range(reuters21578['gnd'].shape[0]):
        topic = reuters21578['gnd'][doc_idx][0]
        if topic_count[topic] < 10:
            to_be_removed_doc_idx.append(doc_idx)
            
    # remove documents with topics that occur less than 10 times
    new_matrix = np.delete(reuters21578['fea'].toarray(), to_be_removed_doc_idx, axis=0)
    new_labels = np.delete(reuters21578['gnd'], to_be_removed_doc_idx, axis=0)
    
    return new_matrix, new_labels


