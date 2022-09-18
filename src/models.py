#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden[0])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden_0 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.layer_hidden_1 = nn.Linear(dim_hidden[1], dim_hidden[2])
        self.layer_hidden_2 = nn.Linear(dim_hidden[2], dim_out)
        # self.layer_hidden_2 = nn.Linear(dim_hidden[2], dim_hidden[3])
        # self.layer_hidden_3 = nn.Linear(dim_hidden[3], dim_out)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden_0(x)
        # x = self.relu(x)
        # x = self.layer_hidden_1(x)
        x = self.relu(x)
        x = self.layer_hidden_2(x)
        # x = self.relu(x)
        # x = self.layer_hidden_3(x)
        return x

    
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2_bn = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2_bn = nn.BatchNorm2d(128)

        self.conv3_1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv3_2_bn = nn.BatchNorm2d(256)

        # self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        # self.conv4_bn = nn.BatchNorm2d(256)
        # self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        # self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc1_bn = nn.BatchNorm1d(256)

        # self.fc2 = nn.Linear(256, 256)
        # self.fc2_bn = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, args.num_classes)

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_1_bn(x)
        x = self.relu(x)
        x = self.conv1_2(x)
        x = self.conv1_2_bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2_1(x)
        x = self.conv2_1_bn(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.conv2_2_bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3_1(x)
        x = self.conv3_1_bn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3_2(x)
        x = self.conv3_2_bn(x)
        x = self.relu(x)
        x = self.pool(x)
        # x = self.dropout1(x)
        
        # x = self.conv4_1(x)
        # x = self.conv4_bn(x)
        # x = self.relu(x)
        # x = self.conv4_2(x)
        # x = self.conv4_bn(x)
        # x = self.relu(x)
        # x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = self.dropout2(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.relu(x)

        # x = self.dropout2(x)
        # x = self.fc2(x)
        # x = self.fc2_bn(x)
        # x = self.relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)

        return x
