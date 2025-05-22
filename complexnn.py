import torch
from torch import nn

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = nn.Linear(in_features, out_features)  # 实部全连接层
        self.fc_i = nn.Linear(in_features, out_features)  # 虚部全连接层

    def forward(self, x):
        a=int(x.shape[1]/2)
        x_r, x_i = torch.split(x, a, dim=1)  # 分离实部和虚部
        out_r = self.fc_r(x_r) - self.fc_i(x_i)  # 计算输出的实部
        out_i = self.fc_r(x_i) + self.fc_i(x_r)  # 计算输出的虚部
        return torch.cat((out_r, out_i), dim=-1)  # 合并实部和虚部

class ComplexAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(ComplexAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            ComplexLinear(input_dim, encoding_dim),
            nn.ReLU(),
            ComplexLinear(encoding_dim, encoding_dim//2),
        )
        self.decoder = nn.Sequential(
            ComplexLinear(encoding_dim//2, encoding_dim),
            nn.ReLU(),
            ComplexLinear(encoding_dim, input_dim),
        )

    def forward(self, x):
        x = x.reshape(-1, 256)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # 将输出数据恢复为2*128的形状
        decoded = decoded.reshape(-1, 2, 128)
        return encoded, decoded
