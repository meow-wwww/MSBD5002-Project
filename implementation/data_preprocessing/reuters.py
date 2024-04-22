# %%
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
from collections import Counter
import typing
import os

NUM_OF_CLASS = 41
FEATURE_DIM = 18933

# %%
def __read_data_core():
    '''
    Read the reuters21578 dataset and return the matrix and labels.
    The returned data has already been preprocessed, so that the dimensionality matches exactly the one used in the paper.
    '''
    print('Reading reuters21578 dataset...')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../dataset/reuters21578.mat')
    reuters21578 = loadmat(data_path) 
    # download from: http://www.cad.zju.edu.cn/home/dengcai/Data/Reuters21578/Reuters21578.mat
    
    print('Counting the number of occurrences of each topic...')
    topic_count = Counter(reuters21578['gnd'].flatten())

    print('Checking documents with topics that occur less than 10 times...')
    to_be_removed_doc_idx = []
    for doc_idx in range(reuters21578['gnd'].shape[0]):
        topic = reuters21578['gnd'][doc_idx][0]
        if topic_count[topic] < 10:
            to_be_removed_doc_idx.append(doc_idx)
            
    print('Removing documents with topics that occur less than 10 times...')
    new_matrix = np.delete(reuters21578['fea'].toarray(), to_be_removed_doc_idx, axis=0)
    new_labels = np.delete(reuters21578['gnd'], to_be_removed_doc_idx, axis=0)
    
    assert new_matrix.shape == (8213, FEATURE_DIM)
    assert new_labels.shape == (8213, 1)
    
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
    assert 0 < num_of_class <= NUM_OF_CLASS, 'num_of_class should be an integer between 1 and 10'
    
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

