#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import tensorly as tl
import tensorly
from itertools import chain
from tensorly.decomposition import parafac


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
        x = self.relu(x)
        x = self.layer_hidden_1(x)
        x = self.relu(x)
        x = self.layer_hidden_2(x)
        # x = self.relu(x)
        # x = self.layer_hidden_3(x)
        return x


class MLP_D(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, rank=4):
        super(MLP_D, self).__init__()
        self.rank = rank
        self.relu = nn.ReLU()

        self.layer_hidden_0_1 = nn.Linear(28, rank)
        self.layer_hidden_0_2 = nn.Linear(28 * rank, rank)
        self.layer_hidden_0_3 = nn.Linear(rank, 32 * rank)
        self.layer_hidden_0_4 = nn.Linear(rank, 32)

        self.layer_hidden_1_1 = nn.Linear(32, rank)
        self.layer_hidden_1_2 = nn.Linear(32 * rank, rank)
        self.layer_hidden_1_3 = nn.Linear(rank, 32 * rank)
        self.layer_hidden_1_4 = nn.Linear(rank, 32)

        self.layer_hidden_2_1 = nn.Linear(32, rank)
        self.layer_hidden_2_2 = nn.Linear(32 * rank, rank)
        self.layer_hidden_2_3 = nn.Linear(rank, 32 * rank)
        self.layer_hidden_2_4 = nn.Linear(rank, 32)

        # self.layer_hidden_3_1 = nn.Linear(32, rank)
        # self.layer_hidden_3_2 = nn.Linear(32 * rank, rank)
        # self.layer_hidden_3_3 = nn.Linear(rank, dim_out)
        self.layer_hidden_3 = nn.Linear(dim_hidden[-1], dim_out)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        number = x.shape[0]
        x = torch.reshape(x, (number, 28, 28))
        x = self.layer_hidden_0_1(x)
        x = torch.reshape(x, (number, 28 * self.rank))
        x = self.layer_hidden_0_2(x)
        x = self.layer_hidden_0_3(x)
        x = torch.reshape(x, (number, 32, self.rank))
        x = self.layer_hidden_0_4(x)
        x = torch.reshape(x, (number, 32 * 32))
        x = self.relu(x)

        x = torch.reshape(x, (number, 32, 32))
        x = self.layer_hidden_1_1(x)
        x = torch.reshape(x, (number, 32 * self.rank))
        x = self.layer_hidden_1_2(x)
        x = self.layer_hidden_1_3(x)
        x = torch.reshape(x, (number, 32, self.rank))
        x = self.layer_hidden_1_4(x)
        x = torch.reshape(x, (number, 32 * 32))
        x = self.relu(x)

        x = torch.reshape(x, (number, 32, 32))
        x = self.layer_hidden_2_1(x)
        x = torch.reshape(x, (number, 32 * self.rank))
        x = self.layer_hidden_2_2(x)
        x = self.layer_hidden_2_3(x)
        x = torch.reshape(x, (number, 32, self.rank))
        x = self.layer_hidden_2_4(x)
        x = torch.reshape(x, (number, 32 * 32))
        x = self.relu(x)

        x = self.layer_hidden_3(x)
        # print(x.size())
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


class CNNFashion_Mnist(nn.Module):
    def __init__(self, args):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


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


class CNNCifar_CP(nn.Module):
    def __init__(self, args, rank):
        super(CNNCifar_CP, self).__init__()
        self.rank = rank

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

        # self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc1_1 = nn.Linear(16 * 2, self.rank)
        self.fc1_2 = nn.Linear(16 * 2 * self.rank, self.rank)
        self.fc1_3 = nn.Linear(self.rank, self.rank * 16)
        self.fc1_4 = nn.Linear(self.rank, 16)
        self.fc1_bn = nn.BatchNorm1d(256)

        # self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, args.num_classes)

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

        x = x.view(x.size(0), -1)
        x = self.dropout2(x)
        # x = self.relu(self.fc1(x))
        # x = self.fc2(x)

        # 128 * 4 * 4
        # 256 * 2 * 2
        number = x.size(0)
        x = torch.reshape(x, (number, 16 * 2, 16 * 2))
        x = self.fc1_1(x)
        x = torch.reshape(x, (number, 16 * 2 * self.rank))
        x = self.fc1_2(x)
        x = self.fc1_3(x)
        x = torch.reshape(x, (number, 16, self.rank))
        x = self.fc1_4(x)
        x = torch.reshape(x, (number, 16 * 16))
        x = self.fc1_bn(x)
        x = self.relu(x)
        
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


def cp_decomposition_conv_layer(layer, rank):
    """ Gets a conv layer and a target rank,
        returns a nn.Sequential object with the decomposition """
    # Perform CP decomposition on the layer weight tensorly.
    # l, f, v, h = parafac(layer.weight.data, rank=rank)[1]
    print(layer.weight.data.size())

    l, f, v, h = parafac(layer.weight.data.numpy(), rank=rank)[1]
    l, f, v, h = torch.from_numpy(l), torch.from_numpy(f), torch.from_numpy(v), torch.from_numpy(h)

    pointwise_s_to_r_layer = torch.nn.Conv2d(
        in_channels=f.shape[0],
        out_channels=f.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False)

    depthwise_vertical_layer = torch.nn.Conv2d(
        in_channels=v.shape[1],
        out_channels=v.shape[1],
        kernel_size=(v.shape[0], 1),
        stride=1, padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        groups=v.shape[1],
        bias=False)

    depthwise_horizontal_layer = torch.nn.Conv2d(
        in_channels=h.shape[1],
        out_channels=h.shape[1],
        kernel_size=(1, h.shape[0]),
        stride=layer.stride,
        padding=(0, layer.padding[0]),
        dilation=layer.dilation,
        groups=h.shape[1],
        bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv2d(
        in_channels=l.shape[1],
        out_channels=l.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data
    depthwise_horizontal_layer.weight.data = torch.transpose(h, 1, 0).unsqueeze(1).unsqueeze(1)
    depthwise_vertical_layer.weight.data = torch.transpose(v, 1, 0).unsqueeze(1).unsqueeze(-1)
    pointwise_s_to_r_layer.weight.data = torch.transpose(f, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = l.unsqueeze(-1).unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer, depthwise_vertical_layer,
                  depthwise_horizontal_layer, pointwise_r_to_t_layer]

    return nn.Sequential(*new_layers)


# decomposition
def decompose(factor=10):
    model = torch.load("cnn_cp").cuda()
    # model = torch.load("cnn_cp")
    model.eval()
    model.cpu()
    layers = model._modules
    for i, key in enumerate(layers.keys()):
        if i >= len(layers.keys()) - 2:
            break
        if isinstance(layers[key], torch.nn.modules.conv.Conv2d):
            conv_layer = layers[key]
            rank = max(conv_layer.weight.data.numpy().shape) // factor
            print('layer:', i, 'rank:', rank)
            layers[key] = cp_decomposition_conv_layer(conv_layer, rank)
        torch.save(model, 'cnn_cp')
    return model
