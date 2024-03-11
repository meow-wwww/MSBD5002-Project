# %%
from scipy.io import loadmat

# %%
def read_data():
    coil20 = loadmat('../../dataset/COIL20.mat')
    # download from: http://www.cad.zju.edu.cn/home/dengcai/Data/COIL20/COIL20.mat
    return coil20['fea'], coil20['gnd'].reshape(-1)


