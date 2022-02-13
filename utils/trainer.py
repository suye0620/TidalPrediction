from tqdm import tqdm
from sklearn.metrics import r2_score

def train(model, loader, criterion, optimizer, config):
    model.train()
    for epoch in range(config.epoch_size):

        for idx, (X, Y) in enumerate(loader):
            X = X.to(config.device)
            Y = Y.to(config.device)
            
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
        X = X.to(config.device)
        Y = Y.to(config.device)
        
        # if idx == 1:
            # print("Y的shape是: ",Y.cpu().shape)
        y_pre += model(X).cpu().squeeze().tolist()
        y += Y.cpu().squeeze().tolist()



    r2Score = r2_score(y_true=y, y_pred=y_pre)
    print("r2Score: ", r2Score)

    # 画出实际结果和预测的结果
    import matplotlib.pyplot as plt
    
    plt.plot(range(len(y[:1000])),y_pre[:1000],color = 'red',linewidth = 1.5,linestyle = '-.',label='prediction')
    plt.plot(range(len(y[:1000])),y[:1000],color = 'blue',linewidth = 1.5,linestyle = '-', label='real')
    plt.legend(loc='best')