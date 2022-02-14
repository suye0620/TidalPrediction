import torch
from torch.utils.data import DataLoader

from config import Config

from utils.dataTools import *
from utils.models import lstm_reg,CNNBiLSTM
from utils.trainer import RNNtrain,evaluate,CNNBiLstmtrain

if __name__ == '__main__':
    config = Config()

    print("Data loading...")
    # 序列数据
    dataset = mydataReader("./dataProcessed/testData.csv")

    # 创建X/Y
    # 划分训练集和测试集，70% 作为训练集
    (train_X ,train_Y ),(test_X ,test_Y )= dataset.split(lookback=config.lookback,trainSet_ratio=0.7)

    # 创建Pytorch使用的dataset
    trainSet = custom_dataset(train_X,train_Y)
    testSet = custom_dataset(test_X,test_Y)

    train_loader = DataLoader(trainSet, batch_size = config.batch_size,
                              shuffle=False, pin_memory=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(testSet, batch_size = config.batch_size,
                            shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    print("Model loading...")
    model = CNNBiLSTM(hidden_size=12,num_layers=2).to(config.device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Training...")
    model = CNNBiLstmtrain(model,
                  loader=train_loader,
                  criterion=criterion,
                  optimizer=optimizer,
                  config=config)

    print("Testing...", round(len(testSet)/config.batch_size))
    evaluate(model, test_loader, config)
