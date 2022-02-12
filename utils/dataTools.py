import numpy as np
import pandas as pd

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