from efficientnet_pytorch import EfficientNet
import torch
from torch.nn import functional as F
from torch import nn
from config import CFG

class CNN(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.map = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.net = EfficientNet.from_name("efficientnet-b0")
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint)
        
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=CFG.cnn_features, bias=True)
    
    def forward(self, x):
        x = F.relu(self.map(x))
        out = self.net(x)
        return out

class Model(nn.Module):
    def __init__(self, cnn_path=None):
        super().__init__()
        self.cnn = CNN(cnn_path)
        self.rnn = nn.LSTM(CFG.cnn_features, CFG.lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(CFG.lstm_hidden, 1, bias=True)

    def forward(self, x):
        # x shape: BxTxCxHxW
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        output, (hn, cn) = self.rnn(r_in)
        
        out = self.fc(hn[-1])
        return out