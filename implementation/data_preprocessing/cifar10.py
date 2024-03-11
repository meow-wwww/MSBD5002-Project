# %%
import pickle
import numpy as np

# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# %%
def read_data():
    cifar10 = unpickle('../../dataset/cifar-10-batches-py/test_batch')
    # download from: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    label = np.array(cifar10[b'labels'])
    label_sort_idx = np.argsort(label)

    sorted_label = label[label_sort_idx]
    assert (np.diff(sorted_label) == 1).sum() == 9
    for i in range(1, 10):
        assert np.diff(sorted_label)[i*1000-1] == 1
    # 10 classes, 1000 images per class
    sorted_data = cifar10[b'data'][label_sort_idx]

    sorted_data_gray = np.zeros((sorted_data.shape[0], 1024))
    for i in range(sorted_data.shape[0]):
        rgb_img = sorted_data[i].reshape(3, 1024)
        gray_img = np.dot(rgb_img.T, [0.299, 0.587, 0.114])
        sorted_data_gray[i] = gray_img
        
    assert sorted_data_gray.shape == (10000, 1024)

    # return 300 images from each class, 
    # here we choose the first 300 images within each class
    return_data = np.zeros((3000, 1024))
    return_label = np.zeros(3000)

    for class_idx in range(10):
        return_data[class_idx*300:(class_idx+1)*300] = sorted_data_gray[class_idx*1000:class_idx*1000+300]
        return_label[class_idx*300:(class_idx+1)*300] = sorted_label[class_idx*1000:class_idx*1000+300]
        
    assert return_data.shape == (3000, 1024)
    assert return_label.shape == (3000,)

    return return_data, return_label


