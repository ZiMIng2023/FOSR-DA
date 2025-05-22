import pickle
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from complexnn import ComplexAutoencoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取您生成的pkl文件，假设文件名为my_data.pkl
with open('./15.pkl', 'rb') as f:
    data = pickle.load(f)
mods=[]
# 获取所有的调制方式和信噪比
mods, snrs = sorted(list(set([k[0] for k in data.keys()]))), sorted(list(set([k[1] for k in data.keys()])))
num_seen_classes=4
# 假设您想要将前三种调制方式作为已知类，后两种调制方式作为未知类
#known_mods = mods[:5]
#unknown_mods = mods[5:]
unknown_mods = [mod for mod in mods if mod in ['comb','single-tone','tiaopin']]
known_mods = [mod for mod in mods if mod not in ['comb', 'tiaopin','single-tone']]
# 将数据按照已知类和未知类划分
known_data = {}
for mod in mods:
    for snr in snrs:
        if mod in known_mods:
            known_data[(mod, snr)] = data[(mod, snr)]
# 设置随机种子
np.random.seed(3417)
# 将已知类的数据转换为numpy数组
X_known = []
y_known = []
for mod in known_mods:
    for snr in snrs:
        X_known.append(known_data[(mod, snr)])# 添加信号数据
for mod in known_mods:
    for snr in snrs:
     for i in range(2500):
      y_known.append((mod, snr)) # 添加标签数据
#y_known=y_known*500
X_known = np.concatenate(X_known,axis=-1) # 将列表转换为二维数组，形状为(2, 128,2000)
X_known = np.transpose(X_known, (2, 0, 1))
#y_known = y_known = np.reshape(y_known, (2000, 2)) # 将列表转换为一维数组，形状为(2000, 2)
y_known = np.array(y_known)
# 将已知类的数据划分为训练集和测试集，假设训练集占70%，测试集占30%
n_examples =X_known.shape[0]
n_train = int(n_examples * 0.8)
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
#训练集
X_train = X_known[train_idx]
y_train = y_known[train_idx]
le = LabelEncoder()
#对标签第一列编码
encoded1 = le.fit_transform(y_train[:, 0])
y_train = np.column_stack((encoded1.astype(str), y_train[:, 1]))
y_train=y_train[:,0]
# 定义一些超参数
batch_size = 200 # 批次大小
# 将numpy数组转换为PyTorch张量
X_source = torch.tensor(X_train).float() # 将训练集信号数据转换为浮点型张量，形状为(105000, 2, 128)
#处理数据的标签,全部换为整型数并取第一列
y_train=y_train.astype(np.int32)
y_source =torch.tensor(y_train)


model = ComplexAutoencoder(input_dim=128, encoding_dim=128)
version='RELEASE'
device = torch.device("cpu")
model_path='./enhancemcom15.pth'.format(version)
#model.load_state_dict(model_path)
model.load_state_dict(torch.load(model_path, map_location=device))
# 将模型设置为评估模式，不进行梯度计算
model.eval()
# 前向传播，得到输出和特征
feature, _ = model(X_source)
# 将特征与原始信号沿着第二个维度拼接，得到3*128的样本
source_combined = torch.cat((X_source, feature.unsqueeze(1)), dim=1)
source_dataset = TensorDataset(source_combined,y_source)
source_loader = DataLoader(source_dataset, batch_size=512, shuffle=True)
num_seen_classes=4
source_map={}
for i in range(num_seen_classes):
  # 根据类别标签从整体样本中筛选出该类别的样本数据
  class_idx = np.where(y_train ==i)
  # 将该类别的样本数据赋值给train_map中对应的键
  source_map[i] = source_combined[class_idx]






