import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_td_loss():
    plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'size': 13}
    # plt.title('Training loss vs Communication rounds', font1)
    file_name = '../save/tt_fc/mlp-local_ep10-train_loss.txt'
    file_name_64 = '../save/tt_fc/mlp_d-tt64-local_ep10-train_loss.txt'
    file_name_32 = '../save/tt_fc/mlp_d-tt32-local_ep10-train_loss.txt'
    file_name_16 = '../save/tt_fc/mlp_d-tt16-local_ep10-train_loss.txt'

    data = np.loadtxt(file_name)
    plt.plot(range(1, len(data[:])+1), data[:], color='r', label='conventional FC', linewidth=2)
    data_64 = np.loadtxt(file_name_64)
    plt.plot(range(1, len(data_64[:])+1), data_64[:], color='blue', label='TT-rank = 64', linewidth=2)
    data_32 = np.loadtxt(file_name_32)
    plt.plot(range(1, len(data_32[:])+1), data_32[:], color='green', label='TT-rank = 32', linewidth=2)
    data_16 = np.loadtxt(file_name_16)
    plt.plot(range(1, len(data_16[:])+1), data_16[:], color='y', label='TT-rank = 16', linewidth=2)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    plt.legend(prop=font2)
    plt.grid()
    plt.show()
    plt.savefig('../save/fc_loss_noiseless.png')


def plot_cnn_loss():
    plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'size': 13}

    file_name = '../save/cnn-local_ep10-train_loss.txt'
    file_name_4 = '../save/cnn_cp-cp4-tt16-local_ep10-train_loss.txt'
    file_name_5 = '../save/cnn_cp-cp5-tt16-local_ep10-train_loss.txt'
    file_name_10 = '../save/cnn_cp-cp10-tt16-local_ep10-train_loss.txt'
    data = np.loadtxt(file_name)
    plt.plot(range(1, len(data[:])+1), data[:], color='r', label='conventional CNN', linewidth=2)
    data_64 = np.loadtxt(file_name_4)
    plt.plot(range(1, len(data_64[:])+1), data_64[:], color='blue', label='CP/TT-1', linewidth=2)
    data_32 = np.loadtxt(file_name_5)
    plt.plot(range(1, len(data_32[:])+1), data_32[:], color='green', label='CP/TT-2', linewidth=2)
    data_16 = np.loadtxt(file_name_10)
    plt.plot(range(1, len(data_16[:])+1), data_16[:], color='y', label='CP/TT-3', linewidth=2)
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    plt.legend(prop=font2)
    plt.grid()
    plt.show()
    plt.savefig('../save/cnn_loss_noiseless.png')


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


def plot_lattice(snr='100'):
    plt.figure()
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 14}
    font2 = {'family': 'Times New Roman',
             'size': 13}

    plt.ylim(ymax=2.5)
    plt.ylim(ymin=-0.1)

    file_name_un_2 = '../save/uncoded_mlp_d/mlp_d-SNR' + snr + '-I2-tt32-local_ep10-train_loss.txt'
    file_name_un_3 = '../save/uncoded_mlp_d/mlp_d-SNR' + snr + '-I3-tt32-local_ep10-train_loss.txt'
    file_name_un_4 = '../save/uncoded_mlp_d/mlp_d-SNR' + snr + '-I4-tt32-local_ep10-train_loss.txt'
    data_un_2 = np.loadtxt(file_name_un_2)
    plt.plot(range(1, len(data_un_2[:])+1), data_un_2[:], color='#FF0000', label='repetition, M=2',
             linewidth=2, zorder=5)
    # data_un_3 = np.loadtxt(file_name_un_3)
    # plt.plot(range(1, len(data_un_3[:])+1), data_un_3[:], color='#D2691E', label='repetition, M=3',
    #          linewidth=2)
    data_un_4 = np.loadtxt(file_name_un_4)
    plt.plot(range(1, len(data_un_4[:])+1), data_un_4[:], color='#D2691E', label='repetition, M=4',
             linewidth=2)  # '#FFD700'

    file_name_un_5 = '../save/uncoded_mlp_d/mlp_d-SNR' + snr + '-I5-tt32-local_ep10-train_loss.txt'
    file_name_un_6 = '../save/uncoded_mlp_d/mlp_d-SNR' + snr + '-I6-tt32-local_ep10-train_loss.txt'
    # data_un_5 = np.loadtxt(file_name_un_5)
    # plt.plot(range(1, len(data_un_5[:]) + 1), data_un_5[:], color='#D2691E', ls='--', label='repetition, M=5',
    #          linewidth=2, zorder=5)
    data_un_6 = np.loadtxt(file_name_un_6)
    plt.plot(range(1, len(data_un_6[:]) + 1), data_un_6[:], color='#FFD700', label='repetition, M=6',
             linewidth=2)


    file_name_l_2 = '../save/lattice_mlp_d/mlp_d-SNR' + snr + '-I2-tt32-local_ep10-train_loss.txt'
    file_name_l_3 = '../save/lattice_mlp_d/mlp_d-SNR' + snr + '-I3-tt32-local_ep10-train_loss.txt'
    file_name_l_4 = '../save/lattice_mlp_d/mlp_d-SNR' + snr + '-I4-tt32-local_ep10-train_loss.txt'
    data_l_2 = np.loadtxt(file_name_l_2)
    plt.plot(range(1, len(data_l_2[:])+1), data_l_2[:], color='#7FFFD4', label='lattice-coded, M=2', linewidth=2)
    # data_l_3 = np.loadtxt(file_name_l_3)
    # plt.plot(range(1, len(data_l_3[:])+1), data_l_3[:], color='#00BFFF', label='lattice-coded, M=3', linewidth=2)
    data_l_4 = np.loadtxt(file_name_l_4)
    plt.plot(range(1, len(data_l_4[:])+1), data_l_4[:], color='#00BFFF', label='lattice-coded, M=4', linewidth=2)

    # file_name_l_5 = '../save/lattice_mlp_d/mlp_d-SNR' + snr + '-I5-tt32-local_ep10-train_loss.txt'
    file_name_l_6 = '../save/lattice_mlp_d/mlp_d-SNR' + snr + '-I6-tt32-local_ep10-train_loss.txt'
    # data_l_5 = np.loadtxt(file_name_l_5)
    # plt.plot(range(1, len(data_l_5[:]) + 1), data_l_5[:], color='#00BFFF', ls='--',
    #          label='lattice-coded, M=5', linewidth=2)
    data_l_6 = np.loadtxt(file_name_l_6)
    plt.plot(range(1, len(data_l_6[:]) + 1), data_l_6[:], color='#0000FF',
             label='lattice-coded, M=6', linewidth=2)

    file_name_noiseless = '../save/tt_fc/mlp_d-tt32-local_ep10-train_loss.txt'
    data_noiseless = np.loadtxt(file_name_noiseless)
    plt.plot(range(1, len(data_noiseless[:]) + 1), data_noiseless[:], color='black', label='noiseless',
             linewidth=2, zorder=8)

    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.ylabel('Training loss', font1)
    plt.xlabel('Communication rounds', font1)
    plt.legend(prop=font2, framealpha=0.3, bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.grid()
    plt.gcf().subplots_adjust(left=0.1, right=0.68)
    plt.show()
    plt.savefig('../save/fc_loss_noisy' + snr + '.png')


def plot_test():
    file_name = '../save/ngas-32-loss.txt'
    data = np.loadtxt(file_name)
    plt.plot(data)
    plt.show()


if __name__ == '__main__':
    # plot_td_loss()
    # plot_ng()
    # plot_test()
    # plot_cnn_loss()
    plot_lattice(snr='3')
    # ng_data_org()
    # plot_ng_new()

    # dif = 6.397173792123795888e-01 - 6.116649031639099787e-01
    # # name = '../save/ng/snr=' + '31' + '/M=' + '2' + '/result/ng-M' + '2' + '-20000epoch.txt'
    # name = '../save/ng-mlp-local_ep1-20000-noiseless-train_loss.txt'
    # data = np.loadtxt(name)
    # for i in range(20000):
    #     data[i] -= (((np.sin((i / 20000 - 1/2) * np.pi) + 1) / 2) ** (1/8) * 0.02)
    # np.savetxt(name, data)

