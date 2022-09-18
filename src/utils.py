#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import copy
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        # apply_transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # first_train_transform = transforms.Compose(
        #     [transforms.RandomHorizontalFlip(p=1),
        #      transforms.ToTensor(),
        #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(32, padding=2),
             transforms.ToTensor(),
             transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transform)
        # train_dataset_flip = datasets.CIFAR10(data_dir, train=True, download=True, transform=first_train_transform)
        # train_dataset = ConcatDataset([train_dataset, train_dataset_flip])

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transform)
        # test_dataset_flip = datasets.CIFAR10(data_dir, train=False, download=True, transform=first_train_transform)
        # test_dataset = ConcatDataset([test_dataset, test_dataset_flip])

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
