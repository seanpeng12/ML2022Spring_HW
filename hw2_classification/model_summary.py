# -*- coding: utf-8 -*-
"""
Defined CNN model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# Model
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



if __name__ == '__main__':
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



    te = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).cuda()
    summary(te, (39 , 41))