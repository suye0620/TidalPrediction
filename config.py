import torch

class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.batch_size = 30
        self.epoch_size = 10
        self.learning_rate = 5e-4
        self.weight_decay = 1e-2

        self.lookback = 10