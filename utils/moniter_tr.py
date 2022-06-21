from os import uname, path
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

if 'rz.uni-augsburg.de' in uname()[1]:
    root = '/User/jx/PycharmProject/gpu5'
else:
    root = '/home/xinjing/Documents/gpu5/home/liushuo/'

filepath = path.abspath(__file__)
slurm = join(root, '../../slurm', '8317_0')
pass


def monitor_tr(slurm):
    score = np.load('utils/cnn14_a3.npy')
    tr_mse = []
    tr_ccc = []

    va_mse = []
    va_ccc = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('##>'):
                elements = line.split(' ')
                if 'Train' in elements[0]:
                    mse = float(elements[4].replace(',', ''))
                    tr_mse.append(mse)
                    ccc = float(elements[9].replace('\n', ''))
                    tr_ccc.append(ccc)
            elif line.startswith('==>'):
                elements = line.split(' ')
                if 'Val' in elements[0]:
                    mse= float(elements[4].replace(',', ''))
                    ccc = float(elements[9].replace('\n', ''))
                    va_mse.append(mse)
                    va_ccc.append(ccc)
    print('max tr ccc:{:2.4f} @:{}'.format(max(tr_ccc), tr_ccc.index(max(tr_ccc))))
    print('max va ccc:{:2.4f} @:{}'.format(max(va_ccc), va_ccc.index(max(va_ccc))))
    plt.figure()
    plt.subplot(121)
    plt.plot(tr_mse)
    plt.plot(va_mse)
    plt.title('ccc_loss')
    plt.legend(['tr_ccc', 'va_ccc'])
    plt.subplot(122)
    plt.plot(tr_ccc)
    plt.plot(va_ccc)
    plt.plot(score)
    plt.title('ccc')
    plt.legend(['tr_ccc', 'va_ccc', 'score'])
    plt.show()


def monitor_L22(slurm):
    score = np.load('utils/cnn14_a3.npy')
    tr_mse = []

    va_mse = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('##>'):
                elements = line.split(' ')
                if 'Train' in elements[0]:
                    mse = float(elements[2].replace(',', ''))
                    tr_mse.append(mse)
            elif line.startswith('==>'):
                elements = line.split(' ')
                if 'Val' in elements[0]:
                    mse= float(elements[2].replace(',', ''))
                    va_mse.append(mse)
    print('max tr ccc:{:2.4f} @:{}'.format(max(tr_mse), tr_mse.index(max(tr_mse))))
    print('max va ccc:{:2.4f} @:{}'.format(max(va_mse), va_mse.index(max(va_mse))))
    plt.figure()
    plt.plot(tr_mse)
    plt.plot(va_mse)
    plt.plot(score)
    plt.title('ccc_s')
    plt.legend(['tr_ccc', 'va_ccc', 'score'])
    plt.show()

def monitor_brl(slurm):
    score = np.load('utils/cnn14_a3.npy')
    tr_ccc = []
    tr_brl = []
    va_ccc = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('##>'):
                elements = line.split(' ')
                if 'Train' in elements[0]:
                    ccc = float(elements[2].replace(',', ''))
                    tr_ccc.append(ccc)
                    brl = float(elements[4].replace('\n', ''))
                    tr_brl.append(brl)
            elif line.startswith('==>'):
                elements = line.split(' ')
                if 'Val' in elements[0]:
                    ccc= float(elements[2].replace(',', ''))
                    va_ccc.append(ccc)
    print('max tr ccc:{:2.4f} @:{}'.format(max(tr_ccc), tr_ccc.index(max(tr_ccc))))
    print('max va ccc:{:2.4f} @:{}'.format(max(va_ccc), va_ccc.index(max(va_ccc))))
    plt.figure()
    plt.subplot(121)
    plt.plot(tr_ccc)
    plt.plot(va_ccc)
    plt.plot(score)
    plt.title('ccc_s')
    plt.legend(['tr_ccc', 'va_ccc', 'score'])
    plt.subplot(122)
    plt.plot(tr_brl)
    plt.title('brl')
    plt.show()

def save_brl(slurm, save_path):
    score = np.load('utils/cnn14_a3.npy')
    tr_ccc = []
    tr_brl = []
    va_ccc = []

    with open(slurm) as slurm_file:
        lines = slurm_file.readlines()
        for line in lines:
            if line.startswith('##>'):
                elements = line.split(' ')
                if 'Train' in elements[0]:
                    ccc = float(elements[2].replace(',', ''))
                    tr_ccc.append(ccc)
                    brl = float(elements[4].replace('\n', ''))
                    tr_brl.append(brl)
                    np.save(save_path, tr_brl)
            elif line.startswith('==>'):
                elements = line.split(' ')
                if 'Val' in elements[0]:
                    ccc= float(elements[2].replace(',', ''))
                    va_ccc.append(ccc)
