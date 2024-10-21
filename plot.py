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
    baseline_valence = 0.8846
    acc_curve_1, sub_list1, sub_arr1 = extract_training_curve(
        'result/0.8/Classifier/DE/with_weight_pseudo_label_normal_online/liking/')
    acc_curve_1 = acc_curve_1[3:]
    # acc_curve_2, sub_list2, sub_arr2 = extract_training_curve(
    #     'result/with_pseudo/with_pretrain/0.8/Classifier/DE/with_weight_pseudo_label_MAML_online_without_meta_pretrain/arousal')
    # acc_curve_2 = acc_curve_2[3:]
    # acc_curve_5, _ = extract_training_curve(
    #     'result/with_pseudo/with_pretrain/0.8/Classifier/DE/with_weight_pseudo_label_MAML_online/arousal/')
    # acc_curve_3, _ = extract_training_curve(
    #     'result/with_pseudo/with_pretrain/0.6/Classifier/PSD/with_weight_pseudo_label_without_metapretrain/valence')
    # acc_curve_4, _ = extract_training_curve(
    #     'result/with_pseudo/with_pretrain/0.8/Classifier/DE/teacher_without_update_Meta_train_without_weight/valence')
    acc_baseline = [baseline_valence for _ in range(len(acc_curve_1))]

    # t_statistic, p_value = stats.ttest_rel(sub_arr1, sub_arr2)

    # 输出t统计量和p值
    # print("t-statistic:", t_statistic)
    # print("p-value:", p_value)

    # sub_list1 = np.array(sub_list1)
    # sub_list2 = np.array(sub_list2)
    #
    # for i in range(sub_list1.shape[1]):
    #     data1 = sub_list1[:, i]
    #     data2 = sub_list2[:, i]
    #     t_test = ttest_rel(data1, data2)

    plt.figure(dpi=300)
    # plt.title('Label: Valence  Features: PSD')
    acc_curve_1 = list(acc_curve_1)
    # acc_curve_1.insert(0, 0.8253)

    # plt.plot(acc_baseline, label='Initial Accuracy', color='red', alpha=0.5)
    plt.plot(acc_curve_1, label='with Meta-pre-train', color='#4e62ab')
    plt.xlabel('Number of arrived online samples (batch)', fontsize=17)
    # plt.tick_params(axis='both', which='major', labelsize=13)
    plt.ylabel('Accuracy (%)', fontsize=17)
    plt.yticks(fontsize=15)
    # plt.xticks([])
    # plt.yticks([])
    # plt.plot(acc_curve_2, label='Without Meta-pre-train', color='red')
    # plt.plot(acc_curve_5, label='DE w/o Meta-pretrain')
    # plt.plot(acc_curve_3, label='PSD w/o Meta-pretrain')
    # plt.plot(acc_curve_4, label='?')
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    plot_acc_curve()