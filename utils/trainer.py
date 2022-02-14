from statistics import mean, mode
from tqdm import tqdm
from numpy import sqrt
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def RNNevaluate(model, loader, config, val_mode=False):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        # batch first,(batch,seq_len,input_size),所以增加第三个维度.因为单序列,所以inputsize维度为1
        X = X.unsqueeze(2).to(config.device)
        Y = Y.to(config.device)
        
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()

    if val_mode:
        valmeanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
        return valmeanSquaredError
    else:
        r2Score = r2_score(y_true=y, y_pred=y_pre)
        meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
        meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
        print("r2Score: ", r2Score)
        print("meanSquaredError: ", meanSquaredError)
        print('RMSE: ',sqrt(meanSquaredError))
        print("meanAbsoluteError: ", meanAbsoluteError)

        # 画出实际结果和预测的结果
        import matplotlib.pyplot as plt
        plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
        plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
        plt.legend(loc='best')

        return (r2Score,meanSquaredError,meanAbsoluteError)

def CNNBiLstm_evaluate(model, loader, config, val_mode=False):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
        X = X.unsqueeze(-2).to(config.device)
        Y = Y.to(config.device)
        if idx == 0:
                print(X.shape)
        
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()

    if val_mode:
        valmeanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
        return valmeanSquaredError
    else:
        r2Score = r2_score(y_true=y, y_pred=y_pre)
        meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
        meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
        print("r2Score: ", r2Score)
        print("meanSquaredError: ", meanSquaredError)
        print('RMSE: ',sqrt(meanSquaredError))
        print("meanAbsoluteError: ", meanAbsoluteError)

        # 画出实际结果和预测的结果
        import matplotlib.pyplot as plt
        plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
        plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
        plt.legend(loc='best')

        return (r2Score,meanSquaredError,meanAbsoluteError)


def RNNtrain(model, trainloader,valloader,criterion, optimizer, config):
    logfile = open("./log/trainLog.txt",mode='a+',encoding='utf-8')
    for epoch in range(config.epoch_size):
        # 每个epoch开始模型训练模式
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            # batch first,(batch,seq_len,input_size),所以增加第三个维度.因为单序列,所以inputsize维度为1
            X = X.unsqueeze(2).to(config.device)
            Y = Y.to(config.device)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")
                
        
        # 一个epoch结束,进行一次验证,并保存相关记录
        valMSE = RNNevaluate(model,loader = valloader,config=config,val_mode=True)
        print(f"Epoch: {epoch} valLoss: {valMSE}")
        logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch,valMSE))
    logfile.write("\n")
    logfile.close()
    return model

def CNNBiLstmtrain(model, trainloader,valloader, criterion, optimizer, config):
    logfile = open("./log/trainLog.txt",mode='a+',encoding='utf-8')
    for epoch in range(config.epoch_size):
        model.train()
        for idx, (X, Y) in enumerate(trainloader):
            # Conv1d接受的数据输入是(batch_size有, channel_size=1, seq_len有),故增加一个通道数，单序列通道数为1，第2维
            X = X.unsqueeze(-2).to(config.device)
            Y = Y.to(config.device)
            if idx == 0:
                print(X.shape)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")
                
        
        # 一个epoch结束,进行一次验证,并保存相关记录
        valMSE = CNNBiLstm_evaluate(model,loader = valloader,config=config,val_mode=True)
        print(f"Epoch: {epoch} valLoss: {valMSE}")
        logfile.write("Epoch: {0} valLoss: {1} \n".format(epoch,valMSE))

    logfile.write("\n")
    logfile.close()
    return model