import torch

# ##############################################################
# 设置神经网络中超参数
# ##############################################################
class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 30
        self.epoch_size = 2
        self.learning_rate = 5e-4
        self.weight_decay = 1e-2
        self.lookback = 10



