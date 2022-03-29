import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
path = '../data/'

# device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

batch_size = 1024
lr = 5e-4
weight_decay = 1e-5
epochs = 100


class MfDataset(Dataset):
    def __init__(self, u_id, i_id, rating):
        self.u_id = u_id
        self.i_id = i_id
        self.rating = rating

    def __getitem__(self, index):
        return self.u_id[index], self.i_id[index], self.rating[index]

    def __len__(self):
        return len(self.rating)


class MF(nn.Module):
    def __init__(self, num_users, num_items, mean, embedding_size=100):
        super(MF, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_size)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_emb = nn.Embedding(num_items, embedding_size)
        self.item_bias = nn.Embedding(num_items, 1)

        self.user_emb.weight.data.uniform_(0, 0.005)
        self.user_bias.weight.data.uniform_(-0.01, 0.01)
        self.item_emb.weight.data.uniform_(0, 0.005)
        self.item_bias.weight.data.uniform_(-0.01, 0.01)

        # 全局bias
        self.mean = nn.Parameter(torch.FloatTensor([mean]), False)

    def forward(self, u_id, i_id):
        U = self.user_emb(u_id)
        b_u = self.user_bias(u_id).squeeze()
        I = self.item_emb(i_id)
        b_i = self.item_bias(i_id).squeeze()
        return torch.mul(U, I).sum(dim=1) + b_u + b_i + self.mean


def main():
    df = pd.read_csv(path + 'u.data', header=None, delimiter='\t')
    x, y = df.iloc[:, :2], df.iloc[:, 2]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 将标签设为np.float32类型
    train_dataset = MfDataset(x_train[0].values, x_train[1].values, y_train.values.astype(np.float32))
    test_dataset = MfDataset(x_test[0].values, x_test[1].values, y_test.values.astype(np.float32))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    mean_rating = df.iloc[:, 2].mean()
    num_users, num_items = max(df[0]) + 1, max(df[1]) + 1
    
    model = MF(num_users, num_items, mean_rating).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):

        model.train()
        total_loss = 0
        for x_u, x_i, y in train_dataloader:
            x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
            
            y_pre = model(x_u, x_i)
            loss = criterion(y_pre, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_dataloader)

        model.eval()
        labels, predicts = [], []
        with torch.no_grad():
            for x_u, x_i, y in test_dataloader:
                x_u, x_i, y = x_u.to(device), x_i.to(device), y.to(device)
                y_pre = model(x_u, x_i)
                labels.extend(y.cpu().numpy().tolist())
                predicts.extend(y_pre.data.cpu().numpy().tolist())
        mse = mean_squared_error(labels, predicts)

        print("Epoch [{}/{}], Loss: {:.4f}, val mse is {:.4f}".format(epoch+1, epochs, train_loss, mse))


if __name__ == '__main__':
    main()
