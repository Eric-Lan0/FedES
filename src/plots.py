import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_ng():
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 13}
    # plt.title('Training loss vs Communication rounds', font1)
    # file_name = '../save/bk_iid_5000rounds_mlp_bacth6000_local-ep1_train-loss.txt'
    # data = np.loadtxt(file_name)
    file_name_as = '../save/es_iid_5000rounds_mlp_bacth64_local-ep1_train-loss.txt'
    data_as = np.loadtxt(file_name_as)
    file_name_sdg = '../save/bk_iid_5000rounds_mlp_bacth6000_local-ep1_train-loss.txt'
    data_sdg = np.loadtxt(file_name_sdg)
    length = 2500
    plt.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x', useMathText=True)
    # plt.plot(range(1, 10 * len(data_as[:])+1, 1), data_as[::1], color='green', label='FedES', linewidth=2)
    plt.plot(range(1, length+1, 1), data_as[:length:1], color='blue', label='FedES', linewidth=2)
    plt.plot(range(1, length+1, 1), data_sdg[:length:1], color='r', label='FedSGD', linewidth=2)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    # plt.legend(prop=font1)
    plt.legend(prop=font2, framealpha=0.5, ncol=1)
    plt.grid()
    plt.show()
    plt.savefig('../save/iid-loss.PNG')


def ng_data_org():
    name_1 = '../save/ng_'
    name_2 = '999mlp-bacth64-local_ep1-train_accuracy.txt'
    for i in range(3):
        if i > 0:
            file_name = name_1 + str(i) + name_2
            data_i = np.loadtxt(file_name)
            data = np.append(data, data_i)
        else:
            file_name = name_1 + name_2
            data = np.loadtxt(file_name)
    name = '../save/es_non-iid_3000rounds_mlp_bacth64_local-ep1_train-accuracy.txt'
    np.savetxt(name, data)


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
