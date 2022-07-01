#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
from options import args_parser
from update_ng import LocalUpdate, test_inference
from models import MLP, MLP_D, CNNMnist, CNNFashion_Mnist, CNNCifar, CNNCifar_CP, decompose
from utils import get_dataset, average_weights, convention_average_weights, lattice_average_weights, exp_details
import matplotlib
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    args = args_parser()
    args.gpu = None  # 'cuda' 'cuda:0' None
    if args.gpu:
        # args.gpu_id = 'cuda:0'
        # torch.cuda.set_device(args.gpu_id)
        device = args.gpu
    else:
        device = 'cpu'

    args.model = 'mlp'  # 'mlp' 'cnn'
    args.dataset = 'mnist'  # 'mnist' 'cifar'
    args.optimizer = 'Adadelta'  # 'RMSprop' 'Adadelta' 'sgd' 'adam'
    args.lr = 0.01  # 0.01 0.005
    args.local_bs = 128
    tt_rank = 32
    tt_rank_cnncifar = 16
    cp_rank = 8
    args.local_ep = 1
    args.epochs = 20000

    num_directions = 1
    std = 0.01

    trans = 'convention'  # 'convention' 'lattice_air'
    SNR = 31.6227766  # 100 (20 dB), 31.6227766 (15 dB), 10 (10 dB), 3.16227766 (5 dB)
    iter = 3

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'cnn_cp':
        global_model = CNNCifar_CP(args=args, rank=tt_rank_cnncifar)
        global_model = global_model.to(device)
        torch.save(global_model, 'cnn_cp')
        decompose()
        global_model = torch.load("cnn_cp").cuda()
        # global_model = torch.load("cnn_cp")

    elif args.model == 'mlp':
        # Multi-layer preceptron
        args.seed = 1
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=[1024, 1024, 1024],
                               dim_out=args.num_classes)
    elif args.model == 'mlp_d':
        # Multi-layer preceptron
        args.seed = 1
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP_D(dim_in=len_in, dim_hidden=[1024, 1024, 1024],
                                 dim_out=args.num_classes, rank=tt_rank)
    else:
        exit('Error: unrecognized model')
    
    start_time = time.time()
    exp_details(args)
    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    # print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        local_walks, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = args.num_users

        for idx in range(idxs_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            walk, loss, length = local_model.update_weights(model=copy.deepcopy(global_model),
                                                            global_round=epoch, std=std, num_directions=num_directions)
            local_walks.append(copy.deepcopy(walk))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        if trans == 'noiseless':
            global_weights = local_model.ng_average_weights(global_model, local_walks, local_losses, length, std)
        elif trans == 'convention':
            global_weights = local_model.ng_convention_average_weights(global_model, local_walks, local_losses,
                                                                       length, std, iter, SNR)
        else:
            exit('Error: unrecognized transmission')

        # update global weights
        global_model.load_state_dict(global_weights)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        loss_avg = sum(list_loss) / len(list_loss)
        train_loss.append(loss_avg)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(loss_avg))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        if (epoch + 1) % 1000 == 0:
            np.savetxt(('../save/ng-epoch_' + trans + str(epoch) + args.model + '-tt_rank' + str(tt_rank)
                        + '-local_ep'
                        + str(args.local_ep) + '-train_loss.txt'), train_loss)
            np.savetxt(('../save/ng-epoch_' + trans + str(epoch) + args.model + '-tt_rank' + str(tt_rank)
                        + '-local_ep'
                        + str(args.local_ep) + '-train_accuracy.txt'), train_accuracy)
            torch.save(global_model, ('ng_tt_epoch' + trans + str(epoch)))
            print('saved: ' + str(epoch))
            train_loss, train_accuracy = [], []

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    np.savetxt(('../save/ng' + args.model + '-SNR' + str(int(SNR)) + '-I' + str(iter) + '-tt_rank' + str(tt_rank)
                + '-local_ep' + str(args.local_ep)
                + '-train_loss.txt'), train_loss)
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.show()
    plt.savefig('../save/ng' + args.model + '-SNR' + str(int(SNR)) + '-I' + str(iter) + '-tt_rank' + str(tt_rank)
                + '-local_ep' + str(args.local_ep) + '-loss.png')

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    np.savetxt(('../save/ng' + args.model + '-SNR' + str(int(SNR)) + '-I' + str(iter) + '-tt_rank' + str(tt_rank)
                + '-local_ep' + str(args.local_ep)
                + '-train_accuracy.txt'), train_accuracy)
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.show()
    plt.savefig('../save/ng' + args.model + '-SNR' + str(int(SNR)) + '-I' + str(iter) + '-tt_rank' + str(tt_rank)
                + '-local_ep' + str(args.local_ep) + '-acc.png')
