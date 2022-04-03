import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import random


def load_data(batch_size):
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]
    col_names = ['label'] + dense_features + sparse_features

    df = pd.read_csv('../data/dac_sample.txt', names=col_names, sep='\t')
    # feature_names = dense_featuress + sparse_featuress

    # 处理缺失值
    df[sparse_features] = df[sparse_features].fillna('-1', )
    df[dense_features] = df[dense_features].fillna(0, )
    # target = ['label']

    # 将类别数据转为数字
    for feat in sparse_features:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    # 将连续值归一化
    mms = MinMaxScaler(feature_range=(0, 1))
    df[dense_features] = mms.fit_transform(df[dense_features])

    feat_size = {}
    feat_size_dense = {feat: 1 for feat in dense_features}
    feat_size_sparse = {
        feat: len(df[feat].unique())
        for feat in sparse_features
    }
    feat_size.update(feat_size_dense)
    feat_size.update(feat_size_sparse)

    feature_columns = [(feat, 'dense') for feat in dense_features
                       ] + [(feat, 'sparse') for feat in sparse_features]

    train, valid = train_test_split(df, test_size=0.1)

    train_dataset = TensorDataset(torch.FloatTensor(train.iloc[:, 1:].values),
                                  torch.FloatTensor(train['label'].values))
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    valid_dataset = TensorDataset(torch.FloatTensor(valid.iloc[:, 1:].values),
                                  torch.FloatTensor(valid['label'].values))
    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False)

    return train_dataset, train_loader, valid_dataset, valid_loader, feat_size, feature_columns


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def accuracy(outputs, labels):
    _, pred = torch.max(outputs.data, dim=1)
    correct = (pred == labels).sum().item()
    return correct / len(labels)
