from dataopen import *
import torch.nn as nn
import torch

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

# 创建自编码器实例
model = ComplexAutoencoder(input_dim=128, encoding_dim=128)
# 定义MSE损失函数
criterion = nn.MSELoss()
# 定义优化器，可以根据需要调整学习率等参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0008)
for epoch in range(300):
    for inputs, labels in train_loader:
        # 假设data是一个批次的输入数据，形状为(batch_size, 2, 128)
        # 前向传播，得到输出
        feature,output = model(inputs)
        # 计算损失
        loss = criterion(output, inputs)
        # 反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印损失值
        print(f"Epoch {epoch}, Loss: {loss.item()}")
torch.save(model.state_dict(), 'enhancem10.pth')





