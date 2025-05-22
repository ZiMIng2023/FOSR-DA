import torch
from source_data import X_source,y_source
from target_data import X_target,y_target
from torch.utils.data import TensorDataset, DataLoader
from complexnn import ComplexAutoencoder

model = ComplexAutoencoder(input_dim=128, encoding_dim=128)
version='RELEASE'
device = torch.device("cpu")
model_path='./enhancemcom15.pth'.format(version)
#model.load_state_dict(model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
# 将模型设置为评估模式，不进行梯度计算
model.eval()
# 前向传播，得到输出和特征
source_feature, _ = model(X_source)
target_feature, _ = model(X_target)

# 将特征与原始信号沿着第二个维度拼接，得到3*128的样本
train_combined = torch.cat((X_source, source_feature.unsqueeze(1)), dim=1)
test_combined = torch.cat((X_target, target_feature.unsqueeze(1)), dim=1)

source_dataset = TensorDataset(train_combined,y_source)
source_loader = DataLoader(source_dataset, batch_size=512, shuffle=True)
target_dataset = TensorDataset(test_combined,y_target)
target_loader = DataLoader(target_dataset, batch_size=512, shuffle=False)



