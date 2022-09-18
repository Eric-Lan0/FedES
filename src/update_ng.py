#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = args.gpu if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader

    def gen_noise(self, net, std):
        noises = []
        for param in net.parameters():
            noises.append(torch.randn_like(param, device=self.device) * std)
        return noises

    def add_noise(self, model, noises):
        for param, noise in zip(model.parameters(), noises):
            param.requires_grad = False
            param += noise
            param.requires_grad = True

    def remove_noise(self, model, noises):
        for param, noise in zip(model.parameters(), noises):
            param.requires_grad = False
            param -= noise
            param.requires_grad = True

    def explore_one_direction(self, net, data, std, if_mirror):
        inputs, targets = data
        noise = self.gen_noise(net, std)
        self.add_noise(net, noise)
        outputs = net(inputs)

        self.remove_noise(net, noise)
        loss = self.criterion(outputs, targets).item()
        if if_mirror:
            # AS method
            inverse_noise = []
            for i in range(len(noise)):
                inverse_noise.append(-noise[i])
            self.add_noise(net, inverse_noise)

            outputs = net(inputs)

            self.remove_noise(net, inverse_noise)
            loss -= self.criterion(outputs, targets).item()
            loss /= 2

        return loss, noise

    def get_es_grad(self, loss_list, noise_list, num, length, std, mode="loss") -> list:
        loss_list = torch.tensor(loss_list, device=self.device)
        # loss_list /= batch_size

        if mode == "loss":
            weight = loss_list
            coefficient = 1
        elif mode == "score":
            weight = 1. / (loss_list + 1e-8)
            coefficient = -1

        # weight /= weight[indices].sum()

        grad = []
        for i in range(len(noise_list[0][0])):
            grad.append(torch.zeros_like(noise_list[0][0][i], device=self.device))

        for n in range(self.args.num_users):
            indices = torch.argsort(weight[n])[-length:]
            for idx in indices:
                for i, g in enumerate(noise_list[n][idx]):
                    grad[i] += coefficient * g * weight[n][idx] / (num * std * std)

        return grad

    def set_es_grad(self, model, grad):
        for param, g in zip(model.parameters(), grad):
            param.grad = g
        return

    def update_weights(self, model, global_round, std):
        # Set mode to train model
        model.train()
        # number = 0

        for iter in range(self.args.local_ep):
            noise_list = []
            loss_list = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                l, noise = self.explore_one_direction(model, (images, labels), std, if_mirror=True)
                loss_list.append(l)
                noise_list.append(noise)

                # model.zero_grad()
                # log_probs = model(images)
                # loss = self.criterion(log_probs, labels)
                # loss.backward()

                # if self.args.verbose and (batch_idx % 10 == 0):
                #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         global_round, iter, batch_idx * len(images),
                #         len(self.trainloader.dataset),
                #         100. * batch_idx / len(self.trainloader), loss.item()))
                # self.logger.add_scalar('loss', loss.item())
                # batch_loss.append(loss.item())
            # epoch_loss.append(sum(batch_loss)/len(batch_loss))
        length = len(loss_list)
        # print(number)
        return noise_list, loss_list, length

    def ng_average_weights(self, model, walks, losses, length, std):
        # Set mode to train model
        model.train()
        epoch_loss = []
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        elif self.args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.args.lr,
                                             weight_decay=1e-4)

        optimizer.zero_grad()
        elite_num = self.args.num_users * length
        grad = self.get_es_grad(losses, walks, elite_num, length, std)
        self.set_es_grad(model, grad)
        optimizer.step()

        return model.state_dict()

    def bk_update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        gradients_epochs = []  # local_epochs, batches, gradients from each batch
        for iter in range(self.args.local_ep):
            gradients = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                gradient = []
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                for parms in model.parameters():
                    # print('parms.grad: ')
                    # print(parms.grad.shape)
                    gradient.append(parms.grad)
                gradients.append(gradient)
            gradients_epochs.append(gradients)
        #     print('gradients', len(gradients))
        # print('gradients_epochs:', len(gradients_epochs))
        local_eps = len(gradients_epochs)
        return gradients_epochs, local_eps

    def bk_average_weights(self, model, gradients, local_eps):
        # Set mode to train model
        model.train()
        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        elif self.args.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'Adadelta':
            optimizer = torch.optim.Adadelta(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        optimizer.zero_grad()
        # elite_num = self.args.num_users * local_eps
        batches = len(gradients[0][0])
        n = batches * local_eps * self.args.num_users
        for idx in range(self.args.num_users):
            for e in range(local_eps):
                for b in range(batches):
                    if idx == 0 and e == 0 and b == 0:
                        grad = copy.deepcopy(gradients[0][0][0])
                        for g in range(len(grad)):
                            if grad[g] is not None:
                                grad[g] /= n
                    else:
                        for g in range(len(gradients[idx][e][b])):
                            if grad[g] is not None:
                                grad[g] += (gradients[idx][e][b][g] / n)

        self.set_es_grad(model, grad)
        optimizer.step()

        return model.state_dict()

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        i = 0

        for batch_idx, (images, labels) in enumerate(self.trainloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()
            i += 1

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
        loss /= i
        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = args.gpu if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    # criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
