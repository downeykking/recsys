import torch
import torch.nn as nn
import torch.optim as optim
import time

from sklearn.metrics import roc_auc_score
from utils import load_data, fix_seed
from model import NFM

# device
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# hyper-parameters
batch_size = 1024
num_epochs = 100
seed = 2022
dropout = 0.3
hidden = 128
lr = 0.001
weight_decay = 0

fix_seed(seed)
train_dataset, train_loader, valid_dataset, valid_loader, feat_size, feature_columns = load_data(
    batch_size=batch_size)

model = NFM(feat_size, feature_columns).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer,
                                      step_size=batch_size,
                                      gamma=0.8)
print(model)


def train(epoch):
    t = time.time()
    model.train()

    global best_auc
    best_auc = 0.0

    total_loss = 0
    train_labels, train_preds = [], []
    for x in train_loader:
        feat, labels = x[0].to(device), x[1].to(device)

        # Forward pass
        outputs = model(feat)
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        train_preds.extend(outputs.data.cpu().numpy().tolist())
        train_labels.extend(labels.cpu().numpy().tolist())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_acc = roc_auc_score(train_labels, train_preds)
    scheduler.step()
    val_acc = valid(model)
    best_auc = val_acc if val_acc > best_auc else best_auc

    print(
        'Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}, Val-Acc: {:.4f}, time: {:.4f}s'
        .format(epoch + 1, num_epochs, total_loss / len(train_loader),
                train_acc, val_acc,
                time.time() - t))


def valid(model):
    model.eval()
    with torch.no_grad():
        valid_labels, valid_preds = [], []
        for x in valid_loader:
            feat, labels = x[0].to(device), x[1].to(device)
            # Forward pass
            outputs = model(feat)
            outputs = outputs.data.cpu().numpy().tolist()
            valid_preds.extend(outputs)
            valid_labels.extend(labels.cpu().numpy().tolist())
        cur_auc = roc_auc_score(valid_labels, valid_preds)
    return cur_auc


# Train model
t_total = time.time()
for epoch in range(num_epochs):
    train(epoch)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s, best auc: {:.4f}".format(
    time.time() - t_total, best_auc))
