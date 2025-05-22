import pickle
import numpy as np
from bianjieyueshu import CenterLoss
from SR2CNN import getSR2CNN
import os
from complexconcate import *
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print(device)
# 定义一些超参数
num_epochs = 300  # 训练轮数
learning_rate = 0.0001  # 学习率
lam_encoder = 1
num_class = 4
feature_dim = 3*128
lam_center = 1
lam_cross = 20
model = getSR2CNN(num_class, feature_dim)

# 定义损失函数和优化器
criterion_encoder = torch.nn.MSELoss()
criterion = nn.CrossEntropyLoss()  #
criterion_cent = CenterLoss(num_class, feat_dim=feature_dim, use_gpu=torch.cuda.is_available())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 使用Adam优化器来更参数
optimizer_cent = torch.optim.Adam(criterion_cent.parameters(), lr=learning_rate)

# 训练自编码器模型
for epoch in range(num_epochs):
    # 训练阶段
    model = model.to(device)
    model.train()  # 将模型设置为训练模式
    train_loss = 0.0  # 初始化训练损失为0
    for inputs, labels in source_loader:  # 遍历训练集中的每个批次
        labels = labels.long()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        a = model.decoder(inputs)
        loss_cross = criterion(outputs, labels)
        loss_cent = criterion_cent(model.getSemantic(inputs), labels)
        loss_encoder = criterion_encoder(model.decoder(inputs), inputs)  # 计算重构输出和原始输入之间的均方误差
        loss = lam_cross * loss_cross + lam_center * loss_cent + lam_encoder * loss_encoder
        optimizer.zero_grad()  # 清空梯度缓存
        loss.backward(retain_graph=True)  # 反向传播计算梯度
        optimizer.step()  # 更新参数
        # train_loss += loss.item() * inputs.size(0) # 累加批次的损失，注意要乘以批次大小
        # train_loss /= len(train_dataset) # 计算整个训练集的平均损失

    print('[%d] loss: %.3f,loss_cross: %.3f,loss_center: %.3f,reconstruction loss: %.3f' % (
    epoch + 1, loss.item(), loss_cross.item(), loss_cent.item(), loss_encoder.item()))

torch.save(model.state_dict(), 'complexmodel10.pth')
