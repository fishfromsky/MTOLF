import argparse
import os
from utils import decompose, normalize, BayesianTorchSolver
import numpy as np
from sklearn.model_selection import train_test_split
from model import MetaLearner, Pretrain_Meta_Learner, OriginalPretain, DNN
import torch
from utils import MetaDataset, shuffle, decompose_DE_dreamer
from torch.utils.data import DataLoader
import copy
import pandas as pd
import scipy.io as sio

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
np.random.seed(100)
torch.random.manual_seed(100)


def main(args):
    valid_acc_sub = []

    for i, sub in enumerate(os.listdir(args.data_dir1)):
        print('processing subject ', sub)
        sub_dir = os.path.join(args.data_dir1, sub)
        data, valence, arousal, dominance, liking = decompose(sub_dir)
        data_norm = normalize(data)
        label = liking

        data_norm, label = shuffle(data_norm, label)

        fold_num = args.fold
        fold_size = data_norm.shape[0] // fold_num
        valid_acc_list = []

        for fold in range(fold_num):
            original_acc = 0
            save_model = None
            print('processing fold ', str(fold))
            valid_index = np.array([i for i in range(fold * fold_size, (fold + 1) * fold_size)])
            total_index = np.array([i for i in range(data_norm.shape[0])])
            train_index = np.array(list(set(total_index) ^ set(valid_index)))

            valid_data = data_norm[valid_index, :]
            valid_label = label[valid_index]
            no_valid_data = data_norm[train_index, :]
            no_valid_label = label[train_index]

            valid_dataset = MetaDataset(valid_data, valid_label)
            valid_db = DataLoader(valid_dataset, batch_size=args.valid_batchsize, shuffle=True, )

            pretrain_data, train_data, pretrain_label, train_label = train_test_split(no_valid_data,
                                                                                      no_valid_label,
                                                                                      test_size=args.ratio_leave_out)

            train_dataset = MetaDataset(train_data, train_label)
            train_db = DataLoader(train_dataset, batch_size=args.mini_batchsize, shuffle=True, drop_last=True)

            base_data, bayesian_data, base_label, bayesian_label = train_test_split(pretrain_data,
                                                                                    pretrain_label,
                                                                                    test_size=args.bayesian_leave_out)

            print('Base sample number: ', len(base_label))
            print('Online sample number: ', len(train_label))

            if args.mode == 'train':
                model_dir = os.path.join(args.pretrained_weight_dir, str(args.ratio_leave_out), str(sub), str(fold))
            else:
                model_dir = os.path.join(args.model_dir, str(sub), str(fold))
            print('Model Directory: ', model_dir)

            if args.mode == 'train':
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                    if args.pretrain_without_meta:
                        pretrained_net = OriginalPretain(base_data.shape[-1], args, base_data, base_label,
                                                         device)
                    else:
                        pretrained_net = Pretrain_Meta_Learner(args, device, base_data, base_label,
                                                               base_data.shape[-1], args.num_cls)
                    pretrained_net.to(device)

                    acc = 0
                    pretrained_model = None

                    for e in range(args.pretrain_epoch):
                        pretrained_net.meta_train()
                        valid_acc = pretrained_net.valid_data(valid_db)
                        if valid_acc > acc:
                            pretrained_model = copy.deepcopy(pretrained_net.net)
                            acc = valid_acc
                        print('Epoch ' + str(e) + ': acc: ', valid_acc)

                    print('Recorded Highest Score: ', acc)

                    torch.save(pretrained_model, os.path.join(model_dir, 'model.pth'))

                else:
                    model_path = os.path.join(model_dir, 'model.pth')
                    dnn_net = torch.load(model_path)

                    solver = BayesianTorchSolver(dnn_net, bayesian_data, bayesian_label, args, device)

                    model = MetaLearner(train_data.shape[-1], valid_db, args, device, base_data, base_label,
                                        solver,
                                        dnn_net)
                    model.to(device)

                    acc_list = []
                    for t, (t_data, t_label) in enumerate(train_db):
                        t_data = t_data.to(device).float()
                        t_label = t_label.to(device).long()
                        if args.normal_online:
                            acc = model.normal_update(t_data)
                        else:
                            acc = model.online_update(t_data)
                        if acc > original_acc:
                            original_acc = acc
                            save_model = copy.deepcopy(model.net)
                        acc_list.append(acc)

                    result_dir = os.path.join(args.save_dir, str(sub), str(fold))
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    model_dir = os.path.join(args.model_dir, str(sub), str(fold))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(save_model, os.path.join(model_dir, 'model.pth'))

                    df = pd.DataFrame({
                        'accuracy': acc_list,
                    })
                    df.to_csv(os.path.join(result_dir, 'result.csv'))

            else:
                model_path = os.path.join(model_dir, 'model.pth')
                dnn_net = torch.load(model_path, map_location='cuda:0')

                model = MetaLearner(train_data.shape[-1], valid_db, args, device, base_data, base_label,
                                    dnn_net)
                model.to(device)

                valid_acc = model.valid(valid_db)
                print(valid_acc)
                valid_acc_list.append(valid_acc)

        if args.mode == 'test':
            valid_acc_sub.append(np.mean(valid_acc_list))

    if args.mode == 'test':
        print('Valid Accuracy: ', np.mean(valid_acc_sub), np.std(valid_acc_sub))


def main_dreamer(args):
    data = sio.loadmat(args.data_dir2)['DREAMER']
    data = data[0]['Data'][0]
    valid_acc_sub = []

    for sub in range(data.shape[1]):

        print('Processing Subject %d' % (sub + 1))
        subject_eeg = data[0, sub][0, 0]
        valence_labels = subject_eeg['ScoreValence']
        arousal_labels = subject_eeg['ScoreArousal']
        dominance_labels = subject_eeg['ScoreDominance']

        valence = valence_labels[:, 0] > 3
        arousal = arousal_labels[:, 0] > 3
        dominance = dominance_labels[:, 0] > 3

        eeg_data = subject_eeg['EEG'][0, 0]
        baseline = eeg_data['baseline']
        stimuli = eeg_data['stimuli']

        total_data, valence, arousal, dominance = decompose_DE_dreamer(stimuli, baseline, valence, arousal, dominance)

        data_norm = normalize(total_data)
        label = arousal

        data_norm, label = shuffle(data_norm, label)

        fold_num = args.fold
        fold_size = data_norm.shape[0] // fold_num
        valid_acc_list = []

        for fold in range(fold_num):
            original_acc = 0
            save_model = None
            print('processing fold ', str(fold))
            valid_index = np.array([i for i in range(fold * fold_size, (fold + 1) * fold_size)])
            total_index = np.array([i for i in range(data_norm.shape[0])])
            train_index = np.array(list(set(total_index) ^ set(valid_index)))

            valid_data = data_norm[valid_index, :]
            valid_label = label[valid_index]
            no_valid_data = data_norm[train_index, :]
            no_valid_label = label[train_index]

            valid_dataset = MetaDataset(valid_data, valid_label)
            valid_db = DataLoader(valid_dataset, batch_size=args.valid_batchsize, shuffle=True, )

            print('Valid Samples: {}'.format(len(valid_label)))

            pretrain_data, train_data, pretrain_label, train_label = train_test_split(no_valid_data,
                                                                                      no_valid_label,
                                                                                      test_size=args.ratio_leave_out)

            train_dataset = MetaDataset(train_data, train_label)
            train_db = DataLoader(train_dataset, batch_size=args.mini_batchsize, shuffle=True, drop_last=True)

            base_data, bayesian_data, base_label, bayesian_label = train_test_split(pretrain_data,
                                                                                    pretrain_label,
                                                                                    test_size=args.bayesian_leave_out)

            print('Base sample number: ', len(base_label))
            print('Online sample number: ', len(train_label))

            if args.mode == 'train':
                model_dir = os.path.join(args.pretrained_weight_dir, str(args.ratio_leave_out), str(sub), str(fold))
            else:
                model_dir = os.path.join(args.model_dir, str(sub), str(fold))
            print('Model Directory: ', model_dir)

            if args.mode == 'train':
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)

                    if args.pretrain_without_meta:
                        pretrained_net = OriginalPretain(base_data.shape[-1], args, base_data, base_label,
                                                         device)
                    else:
                        pretrained_net = Pretrain_Meta_Learner(args, device, base_data, base_label,
                                                               base_data.shape[-1], args.num_cls)
                    pretrained_net.to(device)

                    acc = 0
                    pretrained_model = None

                    for e in range(args.pretrain_epoch):
                        pretrained_net.meta_train()
                        valid_acc = pretrained_net.valid_data(valid_db)
                        if valid_acc > acc:
                            pretrained_model = copy.deepcopy(pretrained_net.net)
                            acc = valid_acc
                        print('Epoch ' + str(e) + ': acc: ', valid_acc)

                    print('Recorded Highest Score: ', acc)

                    torch.save(pretrained_model, os.path.join(model_dir, 'model.pth'))

                else:
                    model_path = os.path.join(model_dir, 'model.pth')
                    dnn_net = torch.load(model_path)

                    solver = BayesianTorchSolver(dnn_net, bayesian_data, bayesian_label, args, device)

                    model = MetaLearner(train_data.shape[-1], valid_db, args, device, base_data, base_label,
                                        solver,
                                        dnn_net)
                    model.to(device)

                    acc_list = []
                    for t, (t_data, t_label) in enumerate(train_db):
                        t_data = t_data.to(device).float()
                        t_label = t_label.to(device).long()
                        acc = model.online_update(t_data)
                        if acc > original_acc:
                            original_acc = acc
                            save_model = copy.deepcopy(model.net)
                        acc_list.append(acc)

                    result_dir = os.path.join(args.save_dir, str(sub), str(fold))
                    if not os.path.exists(result_dir):
                        os.makedirs(result_dir)

                    model_dir = os.path.join(args.model_dir, str(sub), str(fold))
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)

                    torch.save(save_model, os.path.join(model_dir, 'model.pth'))

                    df = pd.DataFrame({
                        'accuracy': acc_list,
                    })
                    df.to_csv(os.path.join(result_dir, 'result.csv'))

            else:
                model_path = os.path.join(model_dir, 'model.pth')
                dnn_net = torch.load(model_path, map_location='cuda:0')

                model = MetaLearner(train_data.shape[-1], valid_db, args, device, base_data, base_label,
                                    dnn_net)
                model.to(device)

                valid_acc = model.valid(valid_db)
                print(valid_acc)
                valid_acc_list.append(valid_acc)

        if args.mode == 'test':
            valid_acc_sub.append(np.mean(valid_acc_list))

    if args.mode == 'test':
        print('Valid Accuracy: ', np.mean(valid_acc_sub), np.std(valid_acc_sub))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG Configuration')

    parser.add_argument('--dataset', default='DEAP', help='dataset to use, "DEAP" or "DREAMER"')

    parser.add_argument('--mode', default='train', help='the model mode')
    parser.add_argument('--data_dir1', default='../GLCNN_Fusion_DEAP/data_preprocessed_matlab/',
                        help='path of the dataset')
    parser.add_argument('--data_dir2', default='F:/EEG数据/DREAMER/DREAMER.mat', help='path of the dataset')

    parser.add_argument('--ratio_leave_out', default=0.8, help='The ratio for leave-out to pretrain')

    parser.add_argument('--bayesian_leave_out', default=0.5, help='the ratio for calculating bayesian')

    parser.add_argument('--save_dir', default='result/with_pseudo/with_pretrain/0.8/Classifier/DE/with_weight_pseudo_label_MAML_normal_online/liking/')
    parser.add_argument('--model_dir', default='model/with_pseudo/with_pretrain/0.8/Classifier/DE/with_weight_pseudo_label_MAML_normal_online/liking/')
    parser.add_argument('--num_cls', default=2, help='the number of categories')
    parser.add_argument('--fold', default=10, help='The number of fold for cross validation')
    parser.add_argument('--update_lr', default=0.001, help='The learning rate to update parameters')
    parser.add_argument('--meta_lr', default=0.01, help='learning rate for mete-train')
    parser.add_argument('--final_batchsize', default=128, help='The finite size of each task')
    parser.add_argument('--mini_batchsize', default=16, help='The size of mini-batches')
    parser.add_argument('--valid_batchsize', default=128, help='The batch size to valid the performance')

    parser.add_argument('--normal_online', default=False, help='If use the normal online learning algorithm')
    parser.add_argument('--use_cluster', default=False, help='if use deep cluster')
    parser.add_argument('--use_weight', default=True, help='If weight samples during training')
    parser.add_argument('--use_bayesian', default=False, help='If use bayesian weight')
    parser.add_argument('--use_meta_online', default=True, help='If use meta online training')

    parser.add_argument('--inner_step', default=5, help='inner step for updating')
    parser.add_argument('--N_update_test', default=10, help='Update_step_of_online')

    parser.add_argument('--threshold', default=0.8, help='The threshold to exclude samples')

    # pretrain meta learning
    parser.add_argument('--support', default=4, help='the number of samples of support samples')
    parser.add_argument('--query', default=12, help='the number of samples of query samples')
    parser.add_argument('--pretrain_update_lr', default=0.001)
    parser.add_argument('--pretrain_meta_lr', default=0.01)
    parser.add_argument('--pretrain_epoch', default=200, help='the epochs for pre-training')
    parser.add_argument('--pretrained_weight_dir', default='pretrained_model/with_meta/liking/DE/')
    parser.add_argument('--pretrain_without_meta', default=False, help='if pretrain model without meta')

    args = parser.parse_args()
    main(args)
    # main_dreamer(args)
