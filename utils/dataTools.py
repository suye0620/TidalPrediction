import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# 定义读取时间序列并进行MMS归一化的dataReader
class mydataReader:
    def __init__(self,filename) -> None:
        data_csv = pd.read_csv(filename, usecols=['time','Water_Level_LAT'],
        index_col='time',parse_dates=['time'])

        # 数据预处理(归一化有利于梯度下降)-MMS标准化
        # 如果不进行归一化，那么由于特征向量中不同特征的取值相差较大，
        # 会导致目标函数变“扁”。这样在进行梯度下降的时候，
        # 梯度的方向就会偏离最小值的方向，走很多弯路，即训练时间过长。
        data_csv = data_csv.dropna()
        max_value = data_csv["Water_Level_LAT"].max()
        min_value = data_csv["Water_Level_LAT"].min()
        scalar = max_value - min_value
        data_csv["Water_Level_LAT"] = data_csv["Water_Level_LAT"].map(lambda x: (x-min_value)/scalar)

        self.data_csv = data_csv

        # 改变值类型
        dataset = data_csv.values
        self.dataset = dataset.astype('float32')

    def split(self,lookback,trainSet_ratio = 0.7):
        # 数据集创建
        dataX, dataY = [], []
        for i in tqdm(range(len(self.dataset) - lookback)):
            a = self.dataset[i:(i + lookback)]
            dataX.append(a)
            dataY.append(self.dataset[i + lookback])
        data_X,data_Y = np.array(dataX), np.array(dataY)

        # 数据划分
        train_size = int(len(data_X) * trainSet_ratio)
        test_size = len(data_X) - train_size
        train_X = data_X[:train_size]
        train_Y = data_Y[:train_size]
        test_X = data_X[train_size:]
        test_Y = data_Y[train_size:]

        print("测试集大小为{}".format(test_size))
        return (train_X,train_Y), (test_X,test_Y)

    def getSeries(self):
        """
        返回原始序列
        """
        return self.data_csv

# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self,data_X,data_Y,lookback):
        """
        :parameters:
        data_X: 构造好的X矩阵
        data_Y: 构造好的Y标签
        最后，我们需要将数据改变一下形状，
        因为 RNN 读入的数据维度是 (seq, batch, feature)，
        所以要重新改变一下数据的维度，这里只有一个序列，
        所以 batch 是 config.batch，而输入的 feature 就是我们希望依据的几个时间步，
        这里我们定的是10，所以 feature 就是 10.
        把feature的维度后调
        """
        data_X = data_X.reshape(-1, 1, lookback)
        data_Y = data_Y.reshape(-1, 1, 1)
        self.X = torch.tensor(data_X, dtype=torch.float32)
        self.Y = torch.tensor(data_Y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)

# 定义一个子类叫 custom_dataset，继承与 Dataset
class CNNdataset(Dataset):
    def __init__(self,data_X,data_Y,lookback):
        """
        :parameters:
        data_X: 构造好的X矩阵
        data_Y: 构造好的Y标签
        最后，我们需要将数据改变一下形状，
        因为 RNN 读入的数据维度是 (seq, batch, feature)，
        所以要重新改变一下数据的维度，这里只有一个序列，
        所以 batch 是 config.batch，而输入的 feature 就是我们希望依据的几个时间步，
        这里我们定的是10，所以 feature 就是 10.
        把feature的维度后调
        """
        self.X = torch.tensor(data_X, dtype=torch.float32)
        self.Y = torch.tensor(data_Y, dtype=torch.float32)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)


"""    废置代码
# 定义读取时间序列并进行MMS归一化的dataReader
def mydataReader(filename):
    data_csv = pd.read_csv(filename, usecols=['time','Water_Level_LAT'],
    index_col='time',parse_dates=['time'])

    # 数据预处理(归一化有利于梯度下降)
    # 如果不进行归一化，那么由于特征向量中不同特征的取值相差较大，
    # 会导致目标函数变“扁”。这样在进行梯度下降的时候，
    # 梯度的方向就会偏离最小值的方向，走很多弯路，即训练时间过长。
    data_csv = data_csv.dropna()
    max_value = data_csv["Water_Level_LAT"].max()
    min_value = data_csv["Water_Level_LAT"].min()
    scalar = max_value - min_value
    data_csv["Water_Level_LAT"] = data_csv["Water_Level_LAT"].map(lambda x: (x-min_value)/scalar)
    
    dataset = data_csv.values
    dataset = dataset.astype('float32')
    return dataset

# supervised-prediction
def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)

# 数据划分
def mydataSplit(data_X,data_Y,trainSet_ratio = 0.7):
    train_size = int(len(data_X) * trainSet_ratio)
    test_size = len(data_X) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    print("测试集大小为{}".format(test_size))
    return (train_X,train_Y), (test_X,test_Y)
"""

