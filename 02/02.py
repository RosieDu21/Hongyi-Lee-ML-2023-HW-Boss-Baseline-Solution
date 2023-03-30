import numpy as np
import torch.nn as nn
import random
import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset

exp_name = 'LSTM_bs64'

# data prarameters
concat_nframes = 1              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.75               # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 1213                      # random seed
batch_size = 64                  # batch size
num_epoch = 150                  # the number of training epoch
learning_rate = 5e-3             # learning rate
lr_decay = 0.965
weight_decay = 1e-4
model_path = f'./{exp_name}_model.ckpt'      # the path where the checkpoint will be saved
early_stopping = 10

# model parameters
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 15               # the number of hidden layers
hidden_dim = 256                 # the hidden dim


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_feat(path):
    feat = torch.load(path)
    return feat


def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)


def concat_feat(x, concat_n):
    assert concat_n % 2 == 1  # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2)  # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid + 1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)


def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, random_seed=1213):
    class_num = 41  # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]

        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(random_seed)
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(
        len(usage_list)))

    X = list()
    if mode == 'train':
        y = list()

    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode == 'train':
            label = torch.LongTensor(label_dict[fname])

        X.append(feat)
        if mode == 'train':
            y.append(label)

    print(f'[INFO] {split} set')
    print(len(X))
    if mode == 'train':
        print(len(y))
        return X, y
    else:
        return X


def padding(batch):
    n = len(batch)
    m = max([len(data[0]) for data in batch])
    X = torch.zeros((n, m) + batch[0][0].shape[1:], dtype=torch.float32)
    mask = torch.zeros((n, m), dtype=torch.bool)
    y = torch.zeros((n, m), dtype=torch.long)
    for i, data in enumerate(batch):
        X[i, :len(data[0]), :] = data[0]
        mask[i, :len(data[0])] = True
        y[i, :len(data[0])] = data[1]
    return X, mask, y


def padding_test(batch):
    n = len(batch)
    m = max([len(data) for data in batch])
    X = torch.zeros((n, m) + batch[0].shape[1:], dtype=torch.float32)
    mask = torch.zeros((n, m), dtype=torch.bool)
    for i, data in enumerate(batch):
        X[i, :len(data), :] = data
        mask[i, :len(data)] = True
    return X, mask


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = y
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


class BaseBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(BaseBlock, self).__init__()

        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.7)

    def forward(self, x):
        x_tmp, h = self.rnn(x)
        x_tmp = nn.functional.relu(self.linear(x_tmp))
        x_tmp = self.bn(x_tmp.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.dropout(x_tmp) + x
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=8, hidden_dim=256):
        super(Classifier, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
        )
        self.rnn = nn.Sequential(
            *[BaseBlock(hidden_dim) for _ in range(hidden_layers)]
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.rnn(x)
        x = self.out(x)
        return x


if __name__=='__main__':
    from torch.utils.data import DataLoader
    import gc

    same_seeds(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'DEVICE: {device}')

    # preprocess data
    train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                       concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)
    val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone',
                                   concat_nframes=concat_nframes, train_ratio=train_ratio, random_seed=seed)

    # get dataset
    train_set = LibriDataset(train_X, train_y)
    val_set = LibriDataset(val_X, val_y)

    # remove raw feature to save memory
    del train_X, train_y, val_X, val_y
    gc.collect()

    # get dataloader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=padding)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=padding)

    # create model, define a loss function, and optimizer
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    best_acc = 0.0
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        train_size = 0
        val_acc = 0.0
        val_loss = 0.0
        val_size = 0

        # training
        model.train()  # set the model to training mode
        for i, batch in enumerate(tqdm(train_loader)):
            features, mask, labels = batch
            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device) * mask

            optimizer.zero_grad()
            outputs = (model(features) * mask.unsqueeze(-1).repeat(1, 1, 41)).permute(0, 2, 1)

            loss = criterion(outputs, labels) / mask.sum()
            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
            train_acc += ((train_pred.detach() == labels.detach()) * mask).sum().item()
            train_loss += loss.item() * mask.sum()
            train_size += mask.sum()

        scheduler.step()

        # validation
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, mask, labels = batch
                features = features.to(device)
                mask = mask.to(device)
                labels = labels.to(device) * mask

                outputs = (model(features) * mask.unsqueeze(-1).repeat(1, 1, 41)).permute(0, 2, 1)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += ((
                                        val_pred.detach() == labels.detach()) * mask).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item()
                val_size += mask.sum()

        print(f'[{epoch + 1:03d}/{num_epoch:03d}] Train Acc: {train_acc / train_size:3.5f} '
              f'Loss: {train_loss / train_size:3.5f} | Val Acc: {val_acc / val_size:3.5f} ' 
              f'loss: {val_loss / val_size:3.5f}')

        # if the model improves, save a checkpoint at this epoch
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f'saving model with acc {best_acc / val_size:.5f}')

    del train_set, val_set
    del train_loader, val_loader
    gc.collect()

    # load data
    test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone',
                             concat_nframes=concat_nframes)
    test_set = LibriDataset(test_X, None)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=padding_test)

    # load model
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(torch.load(model_path))

    pred = np.array([], dtype=np.int32)

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader)):
            features, mask = batch
            features = features.to(device)

            outputs = model(features)

            _, test_pred = torch.max(outputs, 2)  # get the index of the class with the highest probability
            pred = np.concatenate((pred, test_pred.squeeze().cpu().numpy()), axis=0)

    with open('prediction.csv', 'w') as f:
        f.write('Id,Class\n')
        for i, y in enumerate(pred):
            f.write('{},{}\n'.format(i, y))