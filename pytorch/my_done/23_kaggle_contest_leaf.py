from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import functional as F
import os
import time
import platform
import warnings
import numpy as np
import pandas as pd
from PIL import Image

class Accumulator: # 累加器对象
    """ 在 n 个变量上累加 """
    def __init__(self, n):
        self.data = [0.0] * n # python 语法 [0]*n将n个list连接在一起

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        # zip() 将迭代器打包成元组

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class LeafDataset(Dataset):
    """ 自定义数据集 """
    def __init__(self, img_path, labels, has_label=True):
        super().__init__()
        self.has_label = has_label
        self.img_path = img_path
        if self.has_label:
            self.img_labels = torch.tensor(labels, dtype=torch.long)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize([224, 224]),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        if self.has_label:
            label = self.img_labels[index]
            return self.transform(img), label
        else:
            return self.transform(img)
    
    def __len__(self):
        return len(self.img_path)

class Residual(nn.Module):
    # Resnet50 基本结构
    def __init__(self, in_channels, num_channels, stride=1):
        super().__init__()
        self.out_channels = num_channels * 4
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=self.out_channels, kernel_size=1, stride=1) 
        self.conv4 = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        C1 = self.relu(self.bn1(self.conv1(X)))
        C2 = self.relu(self.bn2(self.conv2(C1)))
        C3 = self.bn3(self.conv3(C2))
        C4 = self.conv4(X)
        return self.relu(C3+C4)


def resnet_block(in_channels, num_channels, num_res, is_first=False):
    blk = []
    out_channels = num_channels * 4
    for i in range(num_res):
        if i == 0 and not is_first:
            blk.append(Residual(in_channels=in_channels, num_channels=num_channels, stride=2))
        elif i == 0:
            blk.append(Residual(in_channels=in_channels, num_channels=num_channels, stride=1))
        else:
            blk.append(Residual(in_channels=out_channels, num_channels=num_channels, stride=1))
    return blk

def get_net(num_class):
    # 返回ResNet152 
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # Output 64@56x56
    )
    b2 = nn.Sequential(*resnet_block(64, 64, 3, is_first=True))
    b3 = nn.Sequential(*resnet_block(256, 128, 4, is_first=False))
    b4 = nn.Sequential(*resnet_block(512, 256, 6, is_first=False))
    b5 = nn.Sequential(*resnet_block(1024, 512, 3, is_first=False))

    net = nn.Sequential(b1, b2, b3, b4, b5, 
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(2048, num_class)
    )
    return net

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f"cuda:{i}")
    else:
        return torch.device("cpu")

def get_k_fold_data(k, i, X, y):
    """ 生成交叉验证数据 """
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate([X_train, X_part], axis=0)
            y_train = np.concatenate([y_train, y_part], axis=0)
    return X_train, y_train, X_valid, y_valid

def accuracy(y_hat, y):
    """ 分类问题，统计正确个数 """
    # y_hat 是二维矩阵，取每一行的最大值
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) # 返回最大值对应的序号
    cmp = y_hat.type(y.dtype) == y   # 保证 y 和 y_hat 类型相同
    # cmp 是 bool 类型
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter):
    """使用GPU计算模型在数据集上的精度"""
    device = try_gpu() 
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_gpu(net, train_iter, test_iter, num_epochs, lr):
    scaler = GradScaler()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()
    # 余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=num_epochs,
        eta_min=1e-5
    )
    train_l, train_acc, test_acc, time_l = [], [], [], []
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        start = time.perf_counter()
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                y_hat = net(X)
                l = loss(y_hat, y)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat=y_hat, y=y), X.shape[0])
        scheduler.step()
        end = time.perf_counter()
        train_l.append(metric[0] / metric[2])
        train_acc.append(metric[1] / metric[2])
        test_acc.append(evaluate_accuracy_gpu(net, test_iter))
        time_l.append(end-start)
        if (epoch+1) % 5 == 0:
            print(f"\tEpoch {epoch+1:>2}, Using Time : {time_l[-1]:.4f}, train_acc : {train_acc[-1]:.4f} test_acc : {test_acc[-1]:.4f}")
    return train_l, train_acc, test_acc

def k_fold(k, X_train, y_train, net, num_epochs, learning_rate, batch_size):
    def init_weight(m):
        if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
            torch.nn.init.xavier_uniform_(m.weight) # Pytorch use a=sqrt(5)
    net.apply(init_weight)
    """ K折训练 """
    train_l_sum, train_acc_sum, valid_l_sum = 0, 0, 0
    for i in range(k):
        print(f"折{i+1}:")
        start=time.perf_counter()
        data = get_k_fold_data(k, i, X_train, y_train)
        train_iter, test_iter = [
            torch.utils.data.DataLoader(LeafDataset(data[0].tolist(), data[1].tolist()), batch_size=batch_size, shuffle=True, num_workers=os.cpu_count()),
            torch.utils.data.DataLoader(LeafDataset(data[2].tolist(), data[3].tolist()), batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
        ]
        train_ls, train_acc, valid_ls = train_gpu(net, train_iter, test_iter, num_epochs, learning_rate)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        train_acc_sum += train_acc[-1]
        end=time.perf_counter()
        print()
        print(f'\n折{i + 1}，训练损失:{float(train_ls[-1]):.4f} ,'
              f'训练正确率:{float(train_acc[-1]):.4f} ,'
              f'测试损失:{float(valid_ls[-1]):.4f} ,'
              f'耗时:{end-start:.4f} sec')
        print()
    return train_l_sum / k, train_acc_sum / k, valid_l_sum / k, net


def get_data(train_path, test_path):
    """ 返回训练数据集，测试特征以及映射关系 """
    train_features = pd.read_csv(train_path).image.to_list()
    train_labels = pd.read_csv(train_path).label.to_list()
    test_features = pd.read_csv(test_path).image.to_list()
    train_labels, index = label_map_to_index(train_labels=train_labels)
    
    return train_features, train_labels, test_features, index

def label_map_to_index(train_labels):
    index = list(set(train_labels))
    for i, label in enumerate(train_labels):
        for j in range(len(index)):
            if index[j] == label:
                train_labels[i] = j
    return train_labels, index

def index_map_to_label(y, index):
    for i, label in enumerate(y):
        y[i] = index[int(label)]
    return y

def terminal_init():
    # f = open('a.log', 'a')
    # sys.stdout = f
    warnings.filterwarnings('ignore')
    if platform.system() == 'Linux':
        os.system("clear")
    elif platform.system() == 'Windows':
        os.system("cls")
    print(time.asctime(time.localtime()))


def windows_terminal_wait():
    if platform.system() == 'Windows':
        os.system('pause')

if __name__ == "__main__":

    net_name = 'Resnet-50'
    data_path = os.path.join(".", "data")
    os.chdir(data_path)

    terminal_init()
    all_program_start = time.perf_counter()
    device = try_gpu()


    train_features, train_labels, test_features, index = get_data(
        train_path=os.path.join('train.csv'),
        test_path=os.path.join('test.csv')
    )
    train_features, train_labels, test_features = [np.array(x) for x in (train_features, train_labels, test_features)]

    net = get_net(len(index))
    X = torch.rand(size=(1, 3, 224, 224), dtype=torch.float32)
    print(f"Net({net_name}):")
    for layer in net:
        X = layer(X)
        print(f"\t{layer.__class__.__name__} : {X.shape}")
    print()
    net.to(device)

    k, num_epochs, lr, batch_size = 3, 100, 0.1, 196
    train_l, train_acc, valid_l, net = k_fold(k, train_features, train_labels, net, num_epochs, lr, batch_size)
    print(f'{k}-折验证: 平均训练loss: {float(train_l):.4f}, 平均训练acc: {train_acc:.4f}, 平均验证loss: {float(valid_l):.4f}')
    torch.save(net, os.path.join(".", f"{net_name}_{train_l:.3f}.pkl"))

    # net = torch.load(os.path.join(".", "Resnet-18_0.081.pkl"))
    # net.to(device)

    test_iter = DataLoader(LeafDataset(test_features, None, has_label=False), batch_size=30, shuffle=False, num_workers=os.cpu_count())
    with torch.no_grad():
        out = []
        for X in test_iter:
            X = X.to(device)
            tmp = net(X)
            tmp = tmp.to(torch.device('cpu'))
            out += tmp.argmax(axis=1).detach().tolist()

    out = index_map_to_label(out, index)
    out = np.array(out)
    test = pd.read_csv(os.path.join('.', 'test.csv'))
    test['label'] = pd.Series(out.reshape(1, -1)[0])
    submission = pd.concat([test['image'], test['label']], axis=1)
    submission.to_csv(os.path.join('.', f'{net_name}.csv'), index=False)

    all_program_end = time.perf_counter()
    print(f"Total use {all_program_end - all_program_start:.4f} sec")
    windows_terminal_wait()
