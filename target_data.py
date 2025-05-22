import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 读取您生成的pkl文件，假设文件名为my_data.pkl
with open('./15-raly.pkl', 'rb') as f:
    data = pickle.load(f)
mods=[]
# 获取所有的调制方式和信噪比
mods, snrs = sorted(list(set([k[0] for k in data.keys()]))), sorted(list(set([k[1] for k in data.keys()])))
num_seen_classes=4
#设置已知未知类
unknown_mods = [mod for mod in mods if mod in ['comb', 'single-tone','tiaopin']]
known_mods = [mod for mod in mods if mod not in ['comb', 'single-tone','tiaopin']]
# 将数据按照已知类和未知类划分
known_data = {}
unknown_data = {}
for mod in mods:
    for snr in snrs:
        if mod in known_mods:
            known_data[(mod, snr)] = data[(mod, snr)]
        else:
            unknown_data[(mod, snr)] = data[(mod, snr)]

# 设置随机种子
np.random.seed(417)

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
# 将未知类的数据转换为numpy数组
X_unknown = []
y_unknown = []
for mod in unknown_mods:
    for snr in snrs:
        X_unknown.append(unknown_data[(mod, snr)])  # 添加信号数据

for mod in unknown_mods:
    for snr in snrs:
        for i in range(2500):
            y_unknown.append((mod, snr)) # 添加标签数据
#y_unknown=y_unknown*500
#X_unknown = np.vstack(X_unknown)
X_unknown = np.concatenate(X_unknown,axis=-1)
X_unknown=np.transpose(X_unknown,(2, 0, 1)) # 将列表转换为二维数组，形状为(500, 2, 128)
y_unknown = np.array(y_unknown) # 将列表转换为一维数组，形状为(500, 2)
# 将已知类的数据划分为训练集和测试集，假设训练集占70%，测试集占30%
n_examples =X_known.shape[0]
n_train = int(n_examples * 0.8)
train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
test_idx = list(set(range(0, n_examples)) - set(train_idx))
#训练集
X_train = X_known[train_idx]
y_train = y_known[train_idx]
le = LabelEncoder()
#对标签第一列编码
encoded1 = le.fit_transform(y_train[:, 0])
y_train = np.column_stack((encoded1.astype(str), y_train[:, 1]))
y_train=y_train[:,0]
# 将未知类的数据作为开集测试集
X_zero_test = X_unknown
y_zero_test = y_unknown
n_examples1 =X_zero_test.shape[0]
zero_idx = np.random.choice(range(0, n_examples1), size=1500, replace=False)
X_zero_test = X_zero_test[zero_idx]
y_zero_test = y_zero_test[zero_idx]
#label_dict = {'multi-tone': '5','pulse':'6'}
for i in range(len(y_zero_test)):
    y_zero_test[i][0] = -1
y_zero_test=y_zero_test[:,0]

# 将numpy数组转换为PyTorch张量
X_target = torch.tensor(X_train).float() # 将训练集信号数据转换为浮点型张量，形状为(105000, 2, 128)
X_target_unknown = torch.from_numpy(X_zero_test).float() # 将零样本测试集信号数据转换为浮点型张量，形状为(70000, 2, 128)
#处理数据的标签,全部换为整型数并取第一列
y_train=y_train.astype(np.int32)
y_target =torch.tensor(y_train)
y_zero_test=y_zero_test.astype(np.int32)
y_target_unknown =torch.tensor(y_zero_test)

from complexnn import ComplexAutoencoder

model1 = ComplexAutoencoder(input_dim=128, encoding_dim=128)
version='RELEASE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path1='./enhancemcom15.pth'.format(version)
#model.load_state_dict(model_path)
model1.load_state_dict(torch.load(model_path1, map_location=device))
# 将模型设置为评估模式，不进行梯度计算
model1.eval()
# 前向传播，得到输出和特征
feature, _ = model1(X_target)
z_feature, _ = model1(X_target_unknown)
# 将特征与原始信号沿着第二个维度拼接，得到3*128的样本
target_combined = torch.cat((X_target, feature.unsqueeze(1)), dim=1)
target_unknown_combined = torch.cat((X_target_unknown, z_feature.unsqueeze(1)), dim=1)
num_seen_classes=4
unknown_class=1
target_map={}
for i in range(num_seen_classes):
  # 根据类别标签从整体样本中筛选出该类别的样本数据
  class_idx = np.where(y_target ==i)
  # 将该类别的样本数据赋值给train_map中对应的键
  target_map[i] = target_combined[class_idx]

target_unknown_map={}
for i in range(unknown_class):
    class_idx = np.where(y_target_unknown == -1)
    target_unknown_map[-1] = target_unknown_combined[class_idx]
