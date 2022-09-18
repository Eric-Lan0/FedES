import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_ng():
    plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13}
    # plt.title('Training loss vs Communication rounds', font1)
    file_name = '../save/ng-32-loss.txt'
    file_name_as = '../save/ngas-32-loss.txt'
    file_name_sdg = '../save/mlp_d-tt_rank32-local_ep1-train_loss.txt'
    # file_name = '../save/mlp_d-32-train_loss.txt'
    # file_name_save = '../save/ng-32-loss-0.txt'
    data = np.loadtxt(file_name)
    data_sdg = np.loadtxt(file_name_sdg)
    data_as = np.loadtxt(file_name_as)

    plt.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x', useMathText=True)
    plt.plot(range(0, 10 * len(data_as[:]), 10), data_as[::1], color='green', label='NES', linewidth=2)
    plt.plot(range(0, 10 * len(data[:]), 10), data[::1], color='r', label='NES-AS', linewidth=2)
    plt.plot(range(0, len(data_sdg[:]), 10), data_sdg[::10], color='blue', label='SGD', linewidth=2)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    plt.legend(prop=font1)
    # plt.legend(prop=font1)
    plt.show()
    plt.savefig('../save/ng-32-loss.png')


def ng_data_org():
    snr, M = '31', '1'
    name_1 = '../save/ng/snr=' + snr + '/M=' + M + '/result/ng-epoch_convention'
    name_2 = '999mlp-tt_rank32-local_ep1-train_loss.txt'
    for i in range(20):
        if i > 0:
            file_name = name_1 + str(i) + name_2
            data_i = np.loadtxt(file_name)
            data = np.append(data, data_i)
        else:
            file_name = name_1 + name_2
            data = np.loadtxt(file_name)
    name = '../save/ng/snr=' + snr + '/M=' + M + '/result/ng-M' + M + '-20000epoch.txt'
    np.savetxt(name, data)


def plot_ng_new():
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13}

    snr = ['31']
    M = ['1', '2', '3']
    names = []
    data = []
    for i in range(len(snr)):
        for j in range(len(M)):
            name = '../save/ng/snr=' + snr[i] + '/M=' + M[j] + '/result/ng-M' + M[j] + '-20000epoch.txt'
            data.append(np.loadtxt(name))
            names.append(name)
    name_noiseless = '../save/ng-mlp-local_ep1-20000-noiseless-train_loss.txt'
    data_noiseless = np.loadtxt(name_noiseless)
    plt.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x', useMathText=True)
    plt.plot(range(1, len(data[2])+1), data[2], color='r', label='M=1', linewidth=2)
    plt.plot(range(1, len(data[1]) + 1), data[1], color='green', label='M=2', linewidth=2)
    plt.plot(range(1, len(data[0]) + 1), data[0], color='blue', label='M=3', linewidth=2)
    plt.plot(range(1, len(data_noiseless) + 1), data_noiseless, color='black', label='noiseless', linewidth=2)

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    plt.legend(prop=font2, framealpha=0.5, ncol=1)
    plt.grid()
    # ls='--'
    plt.show()


def plot_test():
    file_name = '../save/ngas-32-loss.txt'
    data = np.loadtxt(file_name)
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    plot_ng()
    # plot_test()
    # ng_data_org()
    # plot_ng_new()
