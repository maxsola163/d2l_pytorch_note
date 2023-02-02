"""
    Kaggle 房价比赛
    MLP + Weight_Decay + Dropout
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
import time
import os
import platform


""" 尝试使用GPU """
def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device("cpu")

def data_read(path='.'):
    """ 数据读取 """
    train_data = pd.read_csv(os.path.join(path, "train.csv"))
    test_data = pd.read_csv(os.path.join(path, "test.csv"))

    """ 数据清洗 删除文字特征 """
    drop_index = [0, 1, 3, 4, 6, 7, 8, 10, 16, 17, 20, 22, 23, 26, 27, 28, 29, 30, 31, 34, 36, 38, 39]
    drop_cols = []
    test = test_data.copy()
    train = train_data.copy()
    for x in drop_index:
        drop_cols.append(list(train_data)[x])
    train_data.drop(columns=drop_cols, inplace=True)
    train_data.drop(columns='Sold Price', inplace=True)
    test_data.drop(columns=drop_cols, inplace=True)

    """ 数字特征处理 归一化 """
    all_features = pd.concat(objs=(train_data, test_data), axis=0)
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)

    """ 数据集生成 """
    n_train = train_data.shape[0]
    train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
    test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
    train_labels = torch.tensor(train['Sold Price'].to_numpy().reshape(-1, 1), dtype=torch.float32)

    return train_features, train_labels, test_features, test


def get_net(train_feature):
    """MLP生成"""
    net = nn.Sequential(
        nn.Linear(train_feature.shape[-1], 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.8),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )
    return net


def log_rmse(net, features, labels, device='cpu'):
    """ 损失函数"""
    features=features.to(device)
    labels=labels.to(device)
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.to(device='cpu').item()


def load_array(data_array, batch_size, is_train=False):
    """ 转换为迭代器 """
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*data_array),
                                       batch_size=batch_size, shuffle=is_train)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """ 训练函数 """
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size, is_train=True)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(try_gpu()), y.to(try_gpu())
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels, try_gpu()))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels, try_gpu()))
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    """ 生成交叉验证数据 """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    """ K折训练 """
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        start=time.perf_counter()
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(train_feature=X_train)
        net.to(device=try_gpu())
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        end=time.perf_counter()
        print(f'折{i + 1}，训练log rmse:{float(train_ls[-1]):f} ,'
              f'验证log rmse:{float(valid_ls[-1]):f} ',
              f'耗时:{end-start:.3f} sec')
    return train_l_sum / k, valid_l_sum / k, net


if __name__ == '__main__':
    
    if platform.system() == "Linux":
        os.system('clear')
    elif platform.system() == "Windows":
        os.system('cls')

    print(f"Start time {time.asctime(time.localtime())}")

    train_features, train_labels, test_features, test = data_read(os.path.join(".", "data"));
    loss = nn.MSELoss()
    
    k, num_epochs, lr, weight_decay, batch_size = 5, 1000, 1, 0.0001, 64
    train_l, valid_l, net = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, 'f'平均验证log rmse: {float(valid_l):f}')
    torch.save(net, os.path.join(".", "data", f"{k}_{num_epochs}_{lr}_{train_l:.3f}.pkl"))

    net.to(torch.device('cpu'))
    preds = net(test_features).detach().numpy()
    test['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test['Id'], test['Sold Price']], axis=1)
    submission.to_csv(os.path.join('.','data', 'submission.csv'), index=False)

    print(log_rmse(net, train_features, train_labels, torch.device('cpu')))
    if platform.system() == 'Windows':
        os.system("pause")
