from tqdm import tqdm
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def RNNtrain(model, loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):
        for idx, (X, Y) in enumerate(loader):
            # 将数据放到GPU上
            """
            最后,我们需要将数据改变一下形状,
            因为 RNN 读入的数据维度是 (seq, batch, feature),
            所以要重新改变一下数据的维度,这里只有一个序列,
            所以 batch 是 config.batch,而输入的 feature 就是我们希望依据的几个时间步,
            这里我们定的是10,所以 feature 就是 10.
            把feature的维度后调
            """
            X = X.reshape(config.lookback,-1,1).to(config.device)
            Y = Y.reshape(1,-1,1).to(config.device)
            if idx == 0:
                print(X.shape)
            
            predict = model(X)
            loss = criterion(predict, Y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx % 10 == 0:
                print(f"Epoch: {epoch} batch: {idx} | loss: {loss}")

    return model

def CNNBiLstmtrain(model, loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):
        for idx, (X, Y) in enumerate(loader):
            # 将数据放到GPU上
            """
            最后,我们需要将数据改变一下形状,
            因为 RNN 读入的数据维度是 (seq, batch, feature),
            所以要重新改变一下数据的维度,这里只有一个序列,
            所以 batch 是 config.batch,而输入的 feature 就是我们希望依据的几个时间步,
            这里我们定的是10,所以 feature 就是 10.
            把feature的维度后调
            """
            X = X.squeeze().unsqueeze(1).to(config.device)
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

    return model

def evaluate(model, loader, config):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        X = X.reshape(config.lookback,-1,1).to(config.device)
        Y = Y.reshape(1,-1,1).to(config.device)
        if idx == 0:
                print(X.shape)
        
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()



    r2Score = r2_score(y_true=y, y_pred=y_pre)
    meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
    meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
    print("r2Score: ", r2Score)
    print("meanSquaredError: ", meanSquaredError)
    print("meanAbsoluteError: ", meanAbsoluteError)

    # 画出实际结果和预测的结果
    import matplotlib.pyplot as plt
    
    plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
    plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
    plt.legend(loc='best')

    return (r2Score,meanSquaredError,meanAbsoluteError)

def CNNBiLstm_evaluate(model, loader, config):
    model.eval()

    y = list()
    y_pre = list()
    for idx, (X, Y) in tqdm(enumerate(loader)):
        X = X.squeeze().unsqueeze(1).to(config.device)
        Y = Y.to(config.device)
        if idx == 0:
                print(X.shape)
        
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()



    r2Score = r2_score(y_true=y, y_pred=y_pre)
    meanSquaredError = mean_squared_error(y_true=y, y_pred=y_pre)
    meanAbsoluteError = mean_absolute_error(y_true=y, y_pred=y_pre)
    print("r2Score: ", r2Score)
    print("meanSquaredError: ", meanSquaredError)
    print("meanAbsoluteError: ", meanAbsoluteError)

    # 画出实际结果和预测的结果
    import matplotlib.pyplot as plt
    
    plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
    plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
    plt.legend(loc='best')

    return (r2Score,meanSquaredError,meanAbsoluteError)