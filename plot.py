import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D
import scipy.stats as stats


def extract_training_curve(file_path, ML=False):
    sub_list = []
    for sub in os.listdir(file_path):
        sub_pth = os.path.join(file_path, sub)
        fold_list = []
        for fold in os.listdir(sub_pth):
            csv_value = pd.read_csv(os.path.join(sub_pth, fold, 'result.csv')).values[:, 1]
            fold_list.append(csv_value)
        fold_acc_curve = sum(fold_list) / len(fold_list)
        sub_list.append(fold_acc_curve)

    sub_list_arr = np.array(sub_list)

    sub_acc_curve = sum(sub_list) / len(sub_list)
    max_id = np.argmax(sub_acc_curve)

    sub_std = np.std(sub_list_arr[:, max_id])

    print(sub_acc_curve[max_id], sub_std)

    if not ML:
        return sub_acc_curve, sub_list, sub_list_arr[:, max_id]
    else:
        return sub_acc_curve, sub_list, sub_list_arr[:, max_id], sub_list_arr[:, 0]


def plot_acc_curve():
    acc_curve_1, sub_list1, sub_arr1 = extract_training_curve(
        'result/0.8/Classifier/DE/with_weight_pseudo_label_normal_online/liking/')

    plt.figure(dpi=300)
    acc_curve_1 = list(acc_curve_1)

    plt.plot(acc_curve_1, label='Thr 0.8 Online', color='#4e62ab')
    plt.xlabel('Number of arrived online samples (batch)', fontsize=17)
    plt.ylabel('Accuracy (%)', fontsize=17)
    plt.yticks(fontsize=15)

    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    plot_acc_curve()