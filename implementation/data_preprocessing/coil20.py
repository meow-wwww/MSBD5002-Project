# %%
from scipy.io import loadmat
import numpy as np
import typing
import random
import os

NUM_OF_CLASS = 20
SAMPLES_PER_CLASS = 72
FEATURE_DIM = 1024

# %%
def __read_data_core():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../dataset/COIL20.mat')
    coil20 = loadmat(data_path)
    # download from: http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat
    
    data = coil20['fea']
    label = coil20['gnd'].reshape(-1)
    assert data.shape == (NUM_OF_CLASS*SAMPLES_PER_CLASS, FEATURE_DIM)
    assert label.shape == (NUM_OF_CLASS*SAMPLES_PER_CLASS,)
    
    return data, label # data and label are sorted by label


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
    assert data_cut.shape == (num_of_class * SAMPLES_PER_CLASS, FEATURE_DIM)
    assert label_cut.shape == (num_of_class * SAMPLES_PER_CLASS,)
    
    # shuffle the data
    if random_shuffle:
        shuffle_idx = np.arange(num_of_class * SAMPLES_PER_CLASS)
        np.random.shuffle(shuffle_idx)
        data_cut = data_cut[shuffle_idx]
        label_cut = label_cut[shuffle_idx]

    return data_cut, label_cut
