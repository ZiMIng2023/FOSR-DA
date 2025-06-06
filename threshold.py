import torch
from dataopen import X_train,y_train
import numpy as np
from complexnn import ComplexAutoencoder


model1 = ComplexAutoencoder(input_dim=128, encoding_dim=128)
version='RELEASE'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path='./enhancemcom15.pth'.format(version)
#model.load_state_dict(model_path)
model1.load_state_dict(torch.load(model_path, map_location=device))
# 将模型设置为评估模式，不进行梯度计算
model1.eval()
# 前向传播，得到输出和特征
feature, _ = model1(X_train)
# 将特征与原始信号沿着第二个维度拼接，得到3*128的样本
train_combined = torch.cat((X_train, feature.unsqueeze(1)), dim=1)
num_seen_classes=4
train_map={}
for i in range(num_seen_classes):
  # 根据类别标签从整体样本中筛选出该类别的样本数据
  class_idx = np.where(y_train ==i)
  # 将该类别的样本数据赋值给train_map中对应的键
  train_map[i] = train_combined[class_idx]

import os
import torch
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from source_data import source_map
from feature_extractor import getSR2CNN

version='RELEASE'
feature_dim=3*128
num_class=4
model = getSR2CNN(num_class,feature_dim)
#model_path='./complexmodel15.pth'.format(version)
model_path='./feature_extractor1.pth'.format(version)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_semantic_ndarray(data):
    tensor_device = torch.Tensor(data).to(device)
    return model.getSemantic(tensor_device).cpu().numpy()


def calculate_distance(x, transform_matrix):
    return np.sqrt(np.dot(np.dot(x, transform_matrix), x.transpose()))


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)

def gen(train_map):
    semantic_center_map = {}
    cov_inv_map = {}
    cov_inv_diag_map = {}
    sigma_identity_map = {}
    distance_map = {}
    for certain_class, train_data in train_map.items():
        raw_output = get_semantic_ndarray(train_data)
        semantic_center_map[certain_class] = np.mean(raw_output, 0)
        covariance_mat = np.cov(raw_output, rowvar=False, bias=True)
        cov_inv_map = np.linalg.pinv(covariance_mat)
        cov_inv_diag_mat = np.diagflat(1 / (covariance_mat.diagonal()))
        cov_inv_diag_mat[cov_inv_diag_mat == np.inf] = 0.0
        cov_inv_diag_map[certain_class] = cov_inv_diag_mat
        sigma = np.mean(np.diagflat(covariance_mat.diagonal()))
        sigma_identity_map[certain_class] = 1 / sigma * np.eye(covariance_mat.shape[0])

    distance_map['Maha'] = cov_inv_map
    distance_map['MahaDiag'] = cov_inv_diag_map
    distance_map['SigmaEye'] = sigma_identity_map

    return semantic_center_map, distance_map

def gen_sematic_vec(train_map,transform_map):
    semantic_center_map = {}
    dist_map_i = []
    dist_map_oushi = []
    dist_map = []
    
    for certain_class, train_data in train_map.items():
        dist_map_i = []
        dists_known = []
        raw_output = get_semantic_ndarray(train_data)
        semantic_center_map[certain_class] = np.mean(raw_output, 0)
        semantic_center = semantic_center_map[certain_class]
        #eyeMat = np.eye(semantic_center_map[certain_class].shape[0])
        for i in range(raw_output.shape[0]):
            #dist_I = calculate_distance(raw_output[i] - semantic_center, eyeMat)
            dist = calculate_distance(raw_output[i] - semantic_center, transform_map[certain_class])
            #dist_map_i.append(dist_I)
            dists_known.append(dist)
        #dist_map_oushi.append(np.array(dist_map_i))
        dist_map.append(np.array(dists_known))
    th = np.zeros(num_class)
    for i in range(len(dist_map)):
        d_i_set = dist_map[i]
        d_i_set = np.sort(d_i_set)[::1]
        mean = np.mean(d_i_set)
        idx = np.ceil(len(d_i_set)*0.95)
        idx = int(idx)
        th[i] = d_i_set[idx]
    return th


model.to(device)

print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))

with torch.no_grad():
    model.eval()
    _, distance_map = gen(train_map)
    distance = 'MahaDiag'
    transform_map = distance_map[distance]
    th = gen_sematic_vec(train_map,transform_map)



