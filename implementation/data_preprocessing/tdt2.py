# %%
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from collections import Counter
import typing
import os

NUM_OF_CLASS = 30
# SAMPLES_PER_CLASS = 72
FEATURE_DIM = 36771

def __read_data_core(top_topics=30):
    '''
    Read the reuters21578 dataset and return the matrix and labels.
    The returned data has already been preprocessed, so that the dimensionality matches exactly the one used in the paper.
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../dataset/TDT2_all.mat')
    tdt2 = loadmat(data_path) 
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
    
    return new_matrix, new_labels.reshape(-1) # data and label are sorted by label


def read_data(num_of_class: int = NUM_OF_CLASS, 
              random_choose_class: typing.Union[bool, typing.List[int]] = False,
              random_shuffle: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
    '''
    Read the cifar10 dataset and return the matrix and labels.

    Parameters:
    num_of_class: int, number of classes to be used, default is 10.
    random_choose_class: bool, whether to randomly choose classes, default is False.
    random_shuffle: bool, whether to randomly shuffle the data, default is False.
    '''
    assert 0 < num_of_class <= NUM_OF_CLASS, f'num_of_class should be an integer between 1 and {NUM_OF_CLASS}'
    
    data, label = __read_data_core()
    
    # randomly choose some classes
    all_distinct_label = np.unique(label)
    assert all_distinct_label.shape == (NUM_OF_CLASS,)
    if random_choose_class == True:
        label_chosen = np.random.choice(all_distinct_label, num_of_class, replace=False)
        label_chosen.sort()
    else:
        if isinstance(random_choose_class, list):
            assert len(random_choose_class) == num_of_class, 'The length of choose_class should be equal to num_of_class'
            assert np.all(np.isin(random_choose_class, all_distinct_label)), 'choose_class should be a subset of all_distinct_label'
            label_chosen = np.array(random_choose_class)
        elif random_choose_class == False:
            label_chosen = all_distinct_label[:num_of_class]
        else:
            raise ValueError('random_choose_class should be a boolean or a list of integers')
    assert np.all(np.diff(label_chosen) > 0)
    
    # cut the corresponding data and label
    chosen_class_index = np.where(np.isin(label, label_chosen))[0]
    data_cut = data[chosen_class_index]
    label_cut = label[chosen_class_index]
    
    # shuffle the data
    if random_shuffle:
        shuffle_idx = np.arange(label_cut.shape[0])
        np.random.shuffle(shuffle_idx)
        data_cut = data_cut[shuffle_idx]
        label_cut = label_cut[shuffle_idx]

    return data_cut, label_cut

