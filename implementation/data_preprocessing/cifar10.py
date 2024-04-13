# %%
import pickle
import numpy as np
import typing
import random, os

NUM_OF_CLASS = 10
SAMPLES_PER_CLASS = 300
FEATURE_DIM = 1024

# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# %%
def read_data_core():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../../dataset/cifar-10-batches-py/test_batch')
    cifar10 = unpickle(data_path)
    # download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    label = np.array(cifar10[b'labels'])
    label_sort_idx = np.argsort(label)

    sorted_label = label[label_sort_idx]
    assert (np.diff(sorted_label) == 1).sum() == NUM_OF_CLASS - 1
    for i in range(1, NUM_OF_CLASS):
        assert np.diff(sorted_label)[i*1000-1] == 1
    # 10 classes, 1000 images per class
    sorted_data = cifar10[b'data'][label_sort_idx]

    sorted_data_gray = np.zeros((sorted_data.shape[0], FEATURE_DIM))
    for i in range(sorted_data.shape[0]):
        rgb_img = sorted_data[i].reshape(3, FEATURE_DIM)
        gray_img = np.dot(rgb_img.T, [0.299, 0.587, 0.114])
        sorted_data_gray[i] = gray_img
        
    assert sorted_data_gray.shape == (10000, FEATURE_DIM)

    # return 300 images from each class, 
    # here we choose the first 300 images within each class
    return_data = np.zeros((NUM_OF_CLASS*SAMPLES_PER_CLASS, FEATURE_DIM))
    return_label = np.zeros(NUM_OF_CLASS*SAMPLES_PER_CLASS).astype(np.uint8)

    for class_idx in range(NUM_OF_CLASS):
        return_data[class_idx*SAMPLES_PER_CLASS:(class_idx+1)*SAMPLES_PER_CLASS] = sorted_data_gray[class_idx*1000:class_idx*1000+SAMPLES_PER_CLASS]
        return_label[class_idx*SAMPLES_PER_CLASS:(class_idx+1)*SAMPLES_PER_CLASS] = sorted_label[class_idx*1000:class_idx*1000+SAMPLES_PER_CLASS]
        
    assert return_data.shape == (NUM_OF_CLASS*SAMPLES_PER_CLASS, FEATURE_DIM)
    assert return_label.shape == (NUM_OF_CLASS*SAMPLES_PER_CLASS,)

    return return_data, return_label # data and label are sorted by label



def read_data(num_of_class: int = 10, random_choose_class: bool = False, random_shuffle: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
    '''
    Read the cifar10 dataset and return the matrix and labels.

    Parameters:
    num_of_class: int, number of classes to be used, default is 10.
    random_choose_class: bool, whether to randomly choose classes, default is False.
    random_shuffle: bool, whether to randomly shuffle the data, default is False.
    '''
    assert 0 < num_of_class <= NUM_OF_CLASS, 'num_of_class should be an integer between 1 and 10'
    
    data, label = read_data_core()
    
    # randomly choose some classes
    all_distinct_label = np.unique(label)
    assert all_distinct_label.shape == (NUM_OF_CLASS,)
    if random_choose_class:
        label_chosen = np.random.choice(all_distinct_label, num_of_class, replace=False)
        label_chosen.sort()
    else:
        label_chosen = all_distinct_label[:num_of_class]
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
