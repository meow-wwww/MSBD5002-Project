# Data Preprocessing

## 实验中用到的数据集下载地址

- coil20: http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat
- cifar-10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- reuters: http://www.cad.zju.edu.cn/home/dengcai/Data/Reuters21578/Reuters21578.mat
- tdt2: http://www.cad.zju.edu.cn/home/dengcai/Data/TDT2/TDT2_all.mat

## API

```import
mat, label = cifar10.load_data()
mat: a numpy array with shape (NumberOfSamples, Dimension)
label: a numpy array with shape (NumberOfSamples, )
```

完整使用实例参见 [./test.ipynb](./test.ipynb)
