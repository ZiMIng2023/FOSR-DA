import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
from feature_extractor import getSR2CNN
import torch
from source_data import source_map
from target_data import target_map,target_unknown_map
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

version='RELEASE'
feature_dim=3*128
num_class=4
model = getSR2CNN(num_class,feature_dim)
#model_path='./complexmodel15.pth'.format(version)
model_path='./complexmodel15.pth'.format(version)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_semantic_ndarray(data):
    tensor_device = torch.Tensor(data).to(device)
    return model.getSemantic(tensor_device).cpu().numpy()


def calculate_distance(x, transform_matrix):
    return np.sqrt(np.dot(np.dot(x, transform_matrix), x.transpose()))


def RESULT_LOGGER(result_list, message):
    result_list.append('{}\n'.format(message))
    print(message)


def gen_sematic_vec(train_map):
    semantic_center_map = {}
    cov_inv_map = {}
    cov_inv_diag_map = {}
    sigma_identity_map = {}
    distance_map = {}

    for certain_class, train_data in train_map.items():
        raw_output = get_semantic_ndarray(train_data)
        semantic_center_map[certain_class] = np.mean(raw_output, 0)

        covariance_mat = np.cov(raw_output, rowvar=False, bias=True)
        #rows,cols=raw_output.shape
        #covariance_mat = np.identity(cols)
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


def classify_evol(transform_map, semantic_center_map, semantic_vector):  #计算距离
    dists_known =[]
    for certain_class in range(num_class):
        semantic_center = semantic_center_map[certain_class]
        dist = calculate_distance(semantic_vector - semantic_center, transform_map[certain_class])
        dists_known.append(dist)
    return dists_known

def cal_acc_evol(train_map, test_map, unknown_test_map, distance='MahaDiag'):#得到所有待测试样本与中心向量的最近距离
    distance_space = []
    semantic_center_map_origin, distance_map = gen_sematic_vec(train_map)
    tackled_test_data = np.concatenate((*(test_map.values()),), 0)
    tackled_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), test_map.items()),), 0)
    tackled_unknown_test_data = np.concatenate((*(unknown_test_map.values()),), 0)
    tackled_unknown_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), unknown_test_map.items()),), 0)
    test_samples = np.concatenate((tackled_test_data, tackled_unknown_test_data), 0)
    test_labels = np.concatenate((tackled_label, tackled_unknown_label), 0)
    predicted_semantics = get_semantic_ndarray(test_samples)
    transform_map = distance_map[distance]
    semanticMap = copy.deepcopy(semantic_center_map_origin)
    for certain_class, predicted_semantic in zip(test_labels, predicted_semantics):
        distance1= classify_evol(transform_map, semanticMap, predicted_semantic)
        distance1 = min(distance1)
        distance_space.append(distance1)
    return distance_space
def find_smallest_quarter_indexes(nums):
    # 将元素和它们的索引组合成一个元组列表
    indexed_nums = list(enumerate(nums))
    # 根据元素值对元组列表进行排序
    indexed_nums.sort(key=lambda x: x[1])
    # 计算要返回的元素数量（列表的25%）
    quarter_length = 2000
    # 获取最小的25%的元素的索引
    smallest_quarter_indexes = [index for index, value in indexed_nums[:quarter_length]]
    return smallest_quarter_indexes
def get_known_target_samples(test_map, unknown_test_map, index):#得到所有待测试样本与中心向量的最近距离
    tackled_test_data = np.concatenate((*(test_map.values()),), 0)
    tackled_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), test_map.items()),), 0)
    tackled_unknown_test_data = np.concatenate((*(unknown_test_map.values()),), 0)
    tackled_unknown_label = np.concatenate(
        (*map(lambda x: np.full([x[1].shape[0]], x[0], dtype=np.int64), unknown_test_map.items()),), 0)
    test_samples = np.concatenate((tackled_test_data, tackled_unknown_test_data), 0)
    test_labels = np.concatenate((tackled_label, tackled_unknown_label), 0)
    known_target_samples = test_samples[index]
    known_target_labels = test_labels[index]
    return known_target_samples,known_target_labels


model.to(device)
print('Loading model from {}'.format(model_path))
model.load_state_dict(torch.load(model_path, map_location=device))
with torch.no_grad():
    model.eval()
    distance_space = cal_acc_evol(source_map, target_map, target_unknown_map)
    target_index = find_smallest_quarter_indexes(distance_space)
    X_target,y_target = get_known_target_samples(target_map,target_unknown_map,target_index)
    X_target = torch.tensor(X_target).float()
    y_target = y_target.astype(np.int32)
    y_target= torch.tensor(y_target)
    target_dataset = TensorDataset(X_target, y_target)
    target_loader = DataLoader(target_dataset, batch_size=512, shuffle=False)
