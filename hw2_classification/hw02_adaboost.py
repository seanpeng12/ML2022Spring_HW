"""### Preparing Data

**Helper functions to pre-process the training data from raw MFCC features of each utterance.**

A phoneme may span several frames and is dependent to past and future frames. \
Hence we concatenate neighboring phonemes for training to achieve higher accuracy. The **concat_feat** function concatenates past and future k frames (total 2k+1 = n frames), and we predict the center frame.

Feel free to modify the data preprocess functions, but **do not drop any frame** (if you modify the functions, remember to check that the number of frames are the same as mentioned in the slides)
"""
from email.policy import strict
import time
import os
import random
import pandas as pd
import torch
from tqdm import tqdm

# use ensemble-pytorch
from torchensemble.utils.logging import set_logger
logger = set_logger('classification_libriDataset_mlp', use_tb_logger=True)

def load_feat(path):
    feat = torch.load(path) # 使用torch.load讀取.pt檔可得到tensor
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

# 可以自己寫 但是不可以drop掉任何frame(必須與ppt frame 數量一致)
def concat_feat(x, concat_n): # 訓練時使用前n個&後n個frame加入一起訓練會更好 可以避免phonene被截到
    # eg. pre. 5 frame & post. 5 frame & self：concat_n = 11(must be odd, 需要對稱)
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y # 每個音樂 X:共有多少個frame  dim:不concat時為39(MFCC) / concat時dim = 39 * n(frame數量)。 y: 對應的label
    else:
      return X

"""## Define Dataset"""

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

"""## Define Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential( # TODO modify stucture of model
            ## 2 layers v1
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 8102),  # can try 16384 (no obvious improvement)
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.BatchNorm1d(8102),
            nn.Linear(8102, output_dim),
            nn.Dropout(p=0.25),
            nn.ReLU(),
        )

        # TODO if boss baseline need RNN rather than sequential
        # 重新寫dataSet且model架構須大改

    def forward(self, x):
        x = self.block(x)
        return x



class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
            BasicBlock(input_dim, hidden_dim),
            *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.fc(x)
        return x



"""## Hyper-parameters"""

# data prarameters
concat_nframes = 1              # TODO the number of frames to concat with, n must be odd (total 2k+1 = n frames) - 原始測資為1 / 因memory不足測試最高只能用到21
train_ratio = 0.8               # the ratio of data used for training, the rest will be used for validation

# training parameter原
seed = 0                        # random seed
batch_size = 512                # batch size - 原始測資為512
num_epoch = 5                   # the number of training epoch
learning_rate = 0.0001          # learning rate
model_path = './VotingClassifier_Classifier_1_ckpt.pth'     # the path where the checkpoint will be saved

# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 1               # the number of hidden layers 原始值為1
hidden_dim = 256               # the hidden dim 原始值為256

"""## Prepare dataset and model"""

import gc

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)
test_set = LibriDataset(test_X, None)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

import numpy as np

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
# Define the ensemble
from torchensemble import VotingClassifier

model = VotingClassifier(
    estimator=Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim),
    n_estimators=1,
    cuda=True,
)
criterion = nn.CrossEntropyLoss() 
model.set_criterion(criterion)
model.set_optimizer('AdamW', lr=learning_rate)
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

records = []

# Train
print("start trainin/testing......")
tic = time.time()
model.fit(
    train_loader,
    epochs=num_epoch,
    test_loader=val_loader,  
    save_model=True
)
toc = time.time()
training_time = toc - tic


# Evaluating/Testing
print("start predict......skip")
# tic = time.time()
# accuracy = model.predict(test_loader)
# toc = time.time()
# evaluating_time = toc - tic
# records.append(("VotingClassifier", training_time, evaluating_time))

del train_loader, val_loader
gc.collect()

# predict
print("start prediction......")
# load model
model = VotingClassifier(
    estimator=Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim),
    n_estimators=1,
    cuda=True,
)
model.load_state_dict(torch.load(model_path))

"""Make prediction."""
test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

"""Write prediction to a CSV file.

After finish running this block, download the file `prediction.csv` from the files section on the left-hand side and submit it to Kaggle.
"""

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))