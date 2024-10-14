import numpy as np
import scipy.io as sio
import math
from scipy.signal import lfilter, butter, welch
from sklearn import preprocessing
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader
import torch
from scipy import stats
import scipy.integrate as itg
import copy
import torch.distributions as dist
import torch.nn.functional as F
import os


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def read_file(file):
    data = sio.loadmat(file)
    data = data['data']

    return data


def read_labels(file):
    label = sio.loadmat(file)['labels']
    return label


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2*math.pi*math.e*variance)/2


def compute_PSD(signal):
    squared_signal = signal**2
    return np.mean(squared_signal)


def decompose(file):
    print('_'*80)
    print('Start reading data ...')
    print('_'*80)
    # trial*channel*sample
    start_index = 384  # 3s pre-trial signals
    data = read_file(file)

    frequency = 128

    decomposed_de = np.empty([0, 4, 60])

    for trial in range(40):

        temp_de = np.empty([0, 60])

        for channel in range(32):
            trial_signal = data[trial, channel, start_index:]
            base_signal = data[trial, channel, :start_index]

            base_theta = butter_bandpass_filter(base_signal, 4, 8, frequency, order=3)
            base_alpha = butter_bandpass_filter(base_signal, 8, 14, frequency, order=3)
            base_beta = butter_bandpass_filter(base_signal, 14, 31, frequency, order=3)
            base_gamma = butter_bandpass_filter(base_signal, 31, 45, frequency, order=3)

            base_theta_DE = 0
            base_alpha_DE = 0
            base_beta_DE = 0
            base_gamma_DE = 0

            for i in range(3):

                base_theta_DE += compute_DE(base_theta[i*frequency:int((i+1)*frequency)])
                base_alpha_DE += compute_DE(base_alpha[i*frequency:int((i+1)*frequency)])
                base_beta_DE += compute_DE(base_beta[i*frequency:int((i+1)*frequency)])
                base_gamma_DE += compute_DE(base_gamma[i*frequency:int((i+1)*frequency)])

            base_beta_DE /= 3
            base_alpha_DE /= 3
            base_beta_DE /= 3
            base_gamma_DE /= 3

            theta = butter_bandpass_filter(trial_signal, 4, 8, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal, 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal, 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal, 31, 45, frequency, order=3)

            DE_theta = np.zeros(shape=[0], dtype=float)
            DE_alpha = np.zeros(shape=[0], dtype=float)
            DE_beta = np.zeros(shape=[0], dtype=float)
            DE_gamma = np.zeros(shape=[0], dtype=float)

            for index in range(60):
                DE_theta = np.append(DE_theta, compute_DE(theta[index*frequency:int((index+1)*frequency)]) - base_theta_DE)
                DE_alpha = np.append(DE_alpha, compute_DE(alpha[index*frequency:int((index+1)*frequency)]) - base_alpha_DE)
                DE_beta = np.append(DE_beta, compute_DE(beta[index*frequency:int((index+1)*frequency)]) - base_beta_DE)
                DE_gamma = np.append(DE_gamma, compute_DE(gamma[index*frequency:int((index+1)*frequency)]) - base_gamma_DE)

            temp_de = np.vstack([temp_de, DE_theta])
            temp_de = np.vstack([temp_de, DE_alpha])
            temp_de = np.vstack([temp_de, DE_beta])
            temp_de = np.vstack([temp_de, DE_gamma])

        temp_trial_de = temp_de.reshape([-1, 4, 60])

        decomposed_de = np.vstack([decomposed_de, temp_trial_de])

    decomposed_de = decomposed_de.reshape([-1, 32, 4, 60]).transpose((0, 3, 2, 1)).reshape([-1, 4, 32]).reshape([-1, 128])

    labels = read_labels(file)
    valence_labels = labels[:, 0] > 5
    arousal_labels = labels[:, 1] > 5
    dominance_labels = labels[:, 2] > 5
    liking_labels = labels[:, 3] > 5
    valence_labels = valence_labels.astype(np.int)
    arousal_labels = arousal_labels.astype(np.int)
    dominance_labels = dominance_labels.astype(np.int)
    liking_labels = liking_labels.astype(np.int)
    valence = []
    arousal = []
    dominance = []
    liking = []
    for i in range(len(valence_labels)):
        valence.extend([valence_labels[i] for _ in range(60)])
        arousal.extend([arousal_labels[i] for _ in range(60)])
        dominance.extend([dominance_labels[i] for _ in range(60)])
        liking.extend([liking_labels[i] for _ in range(60)])

    return decomposed_de, np.array(valence), np.array(arousal), np.array(dominance), np.array(liking)


def decompose_DE_dreamer(stimuli, baseline, valence, arousal, dominance):
    frequency = 128

    total_time_data = np.empty([0, 56])

    total_valence_labels = []
    total_arousal_labels = []
    total_dominance_labels = []

    total_data_DE = np.empty([0, 56])

    for trial in range(18):

        num_sample = stimuli[trial][0].shape[0] // frequency
        temp_de = np.empty([0, num_sample])

        for channel in range(14):

            channel_data_baseline = baseline[trial][0][:, channel]
            channel_data_stimuli = stimuli[trial][0][:, channel]

            baseline_theta = butter_bandpass_filter(channel_data_baseline, 4, 8, frequency, order=3)
            baseline_alpha = butter_bandpass_filter(channel_data_baseline, 8, 14, frequency, order=3)
            baseline_beta = butter_bandpass_filter(channel_data_baseline, 14, 31, frequency, order=3)
            baseline_gamma = butter_bandpass_filter(channel_data_baseline, 31, 45, frequency, order=3)

            baseline_theta_DE = []
            baseline_alpha_DE = []
            baseline_beta_DE = []
            baseline_gamma_DE = []

            for i in range(60):
                baseline_beta_DE.append(compute_DE(baseline_beta[i * frequency:(i + 1) * frequency]))
                baseline_theta_DE.append(compute_DE(baseline_theta[i * frequency:(i + 1) * frequency]))
                baseline_alpha_DE.append(compute_DE(baseline_alpha[i * frequency:(i + 1) * frequency]))
                baseline_gamma_DE.append(compute_DE(baseline_gamma[i * frequency:(i + 1) * frequency]))

            baseline_theta_DE_avg = np.mean(baseline_theta_DE)
            baseline_alpha_DE_avg = np.mean(baseline_alpha_DE)
            baseline_beta_DE_avg = np.mean(baseline_beta_DE)
            baseline_gamma_DE_avg = np.mean(baseline_gamma_DE)

            stimuli_theta = butter_bandpass_filter(channel_data_stimuli, 4, 8, frequency, order=3)
            stimuli_alpha = butter_bandpass_filter(channel_data_stimuli, 8, 14, frequency, order=3)
            stimuli_beta = butter_bandpass_filter(channel_data_stimuli, 14, 31, frequency, order=3)
            stimuli_gamma = butter_bandpass_filter(channel_data_stimuli, 31, 45, frequency, order=3)

            stimuli_DE_theta = np.empty([0])
            stimuli_DE_alpha = np.empty([0])
            stimuli_DE_beta = np.empty([0])
            stimuli_DE_gamma = np.empty([0])

            for i in range(num_sample):
                stimuli_DE_theta = np.append(stimuli_DE_theta,
                                             compute_DE(stimuli_theta[i * frequency:(i + 1) * frequency])
                                             - baseline_theta_DE_avg)
                stimuli_DE_alpha = np.append(stimuli_DE_alpha,
                                             compute_DE(stimuli_alpha[i * frequency:(i + 1) * frequency])
                                             - baseline_alpha_DE_avg)
                stimuli_DE_beta = np.append(stimuli_DE_beta, compute_DE(stimuli_beta[i * frequency:(i + 1) * frequency])
                                            - baseline_beta_DE_avg)
                stimuli_DE_gamma = np.append(stimuli_DE_gamma,
                                             compute_DE(stimuli_gamma[i * frequency:(i + 1) * frequency])
                                             - baseline_gamma_DE_avg)

            temp_de = np.vstack([temp_de, stimuli_DE_theta])
            temp_de = np.vstack([temp_de, stimuli_DE_alpha])
            temp_de = np.vstack([temp_de, stimuli_DE_beta])
            temp_de = np.vstack([temp_de, stimuli_DE_gamma])

        temp_trial_DE = temp_de.reshape([-1, 4, num_sample])
        temp_trial_DE = temp_trial_DE.transpose((2, 0, 1)).reshape([-1, 56])

        total_data_DE = np.vstack([total_data_DE, temp_trial_DE])
        total_valence_labels.extend([valence[trial] for _ in range(temp_trial_DE.shape[0])])
        total_arousal_labels.extend([arousal[trial] for _ in range(temp_trial_DE.shape[0])])
        total_dominance_labels.extend([dominance[trial] for _ in range(temp_trial_DE.shape[0])])

    total_valence_labels = np.array(total_valence_labels)
    total_arousal_labels = np.array(total_arousal_labels)
    total_dominance_labels = np.array(total_dominance_labels)

    return total_data_DE, total_valence_labels, total_arousal_labels, total_dominance_labels


def min_max_norm(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normalize(data):
    data = preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=True)
    # data = data.T
    # for i in range(data.shape[0]):
    #     data[i] = min_max_norm(data[i, :])
    # data = data.T
    return data


def shuffle(data_norm, label):
    indexes = np.array([i for i in range(data_norm.shape[0])])
    np.random.shuffle(indexes)
    data_norm = data_norm[indexes, :]
    label = label[indexes]

    return data_norm, label


class MetaDataset(Dataset):
    def __init__(self, data, label):
        super(MetaDataset, self).__init__()

        self.data = data
        self.label = label

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.label)


class BayesianSolver:
    def __init__(self, model, valid_data, valid_label, args):

        self.valid_data = valid_data
        self.valid_label = valid_label
        self.model = model
        self.args = args

        self.param_list, self.prior_list = self.calculate_pdf()

    def update_pdf(self, model):
        self.model = model
        self.param_list, self.prior_list = self.calculate_pdf()

    def calculate_pdf(self):
        params_list, prior_list = [], []
        real_label = self.valid_label
        pred_prob = self.model.predict_proba(self.valid_data)
        for i in range(self.args.num_cls):
            prob_set = pred_prob[real_label == i][:, i]
            distribution = stats.norm

            params = distribution.fit(prob_set)
            params_list.append(params)

            prior = sum(real_label == i) / len(real_label)
            prior_list.append(prior)

        return params_list, prior_list

    def calculate_prob(self, X):
        bayesian_weight = []
        for i in range(X.shape[0]):
            likelihood_list = []
            for j in range(self.args.num_cls):
                likelihood = itg.quad(stats.norm.pdf, float('-Inf'), X[i, j], args=self.param_list[j])[0]  # P(X|Y)
                likelihood_list.append(likelihood)

            evidence = sum([likelihood_list[label]*self.prior_list[label] for label in range(len(self.prior_list))])
            posterior = np.array([(likelihood_list[label]*self.prior_list[label])/(evidence+1e-4) for label in range(len(self.prior_list))])
            bayesian_weight.append(posterior)

        return np.array(bayesian_weight)


class BayesianTorchSolver:
    def __init__(self, model, bayesian_data, bayesian_label, args, device):
        self.model = copy.deepcopy(model)
        self.bayesian_data = torch.from_numpy(bayesian_data).to(device).float()
        self.bayesian_label = torch.from_numpy(bayesian_label).to(device).long()
        self.args = args
        self.device = device

        self.prior_list, self.params = self.calculate_pdf()

    def update(self, model):
        self.model = copy.deepcopy(model)
        self.calculate_pdf()

    def calculate_pdf(self):
        with torch.no_grad():
            params = []
            prior_list = []
            self.model.eval()
            pred_prob = F.softmax(self.model(self.bayesian_data), dim=-1)
            for i in range(self.args.num_cls):
                pred_slice = pred_prob[self.bayesian_label == i, i]
                param1 = pred_slice.mean(dim=0)
                param2 = pred_slice.std(dim=0)
                params.append((param1, param2))

                prior = torch.sum(self.bayesian_label == i) / len(self.bayesian_label)
                prior_list.append(prior)

            prior_list = torch.tensor(prior_list).to(self.device).float()
            params = params

        return prior_list, params

    def calculate_prob(self, X):
        X = F.softmax(X, dim=-1)
        posterior_list = torch.zeros((X.shape[0], self.args.num_cls)).to(self.device).float()
        for i in range(X.shape[0]):
            data_slice = X[i, :]
            likelihood_list = torch.zeros((self.args.num_cls,)).to(self.device).float()
            for j in range(self.args.num_cls):
                upper_bound = data_slice[j]
                norm_dist = dist.Normal(self.params[j][0], self.params[j][1])
                condition1 = norm_dist.cdf(upper_bound)
                likelihood_list[j] = condition1

            evidence = torch.matmul(likelihood_list.unsqueeze(0), self.prior_list.unsqueeze(1)).squeeze()
            for j in range(self.args.num_cls):
                posterior_list[i, j] = (likelihood_list[j] * self.prior_list[j]) / evidence

        return posterior_list


def delete_list_by_index(list_given, index_to_delete):
    for counter, index in enumerate(index_to_delete):
        index = index - counter
        list_given.pop(index)

    return list_given


def select_list_by_index(list_given, index_to_select):
    select_list = []
    for idx in index_to_select:
        select_list.append(list_given[idx])

    return select_list


def gaussian_kernel(x, y, sigma=1.0):
    """Compute the Gaussian (RBF) kernel between arrays x and y."""
    dists = cdist(x, y, 'sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))


def compute_mmd(x, y, sigma=1.0):
    """Compute the Maximum Mean Discrepancy (MMD) between samples x and y."""
    xx_kernel = gaussian_kernel(x, x, sigma)
    yy_kernel = gaussian_kernel(y, y, sigma)
    xy_kernel = gaussian_kernel(x, y, sigma)

    mmd = np.mean(xx_kernel) + np.mean(yy_kernel) - 2 * np.mean(xy_kernel)
    return mmd


class ShadowModel:
    def __init__(self, base_model, base_data, base_label, valid_data, valid_label, args, solver):

        self.model = base_model
        self.valid_data = valid_data
        self.valid_label = valid_label
        self.base_data = base_data
        self.base_label = base_label
        self.base_weight = np.array([1 for _ in range(len(self.base_label))])

        self.data_buffer = []
        self.label_buffer = []
        self.weight_buffer = []
        self.original_acc = 0

        self.solver = solver

        self.args = args

        self.select_buffer_num = 10  # 取根据mmd距离排序前10个

    def pretrain(self):
        self.model.fit(self.base_data, self.base_label)
        self.original_acc = self.valid(self.model)
        print('Original Acc: ', self.original_acc)
        return self.original_acc

    def valid(self, model):
        valid_logits = model.predict(self.valid_data)
        original_acc = np.sum(valid_logits == self.valid_label) / len(self.valid_label)
        return original_acc

    def online_update(self, data_slice):
        self.data_buffer.append(data_slice)

        if not self.args.use_weight:
            label_slice = self.model.predict(data_slice)
            self.label_buffer.append(label_slice)
            prob_slice = np.array([1 for _ in range(len(label_slice))])
        else:
            if self.args.use_decision:
                label_prob = self.model.predict_proba(data_slice)
                label_slice = np.argmax(label_prob, axis=-1)
                self.label_buffer.append(label_slice)
                prob_slice = np.array([label_prob[i, label_slice[i]] for i in range(len(label_slice))])
            else:
                label_prob = self.solver.calculate_prob(data_slice)
                label_slice = np.argmax(label_prob, axis=-1)
                self.label_buffer.append(label_slice)
                prob_slice = np.array([label_prob[i, label_slice[i]] for i in range(len(label_slice))])

        self.weight_buffer.append(prob_slice)

        mmd_score = [compute_mmd(self.base_data, self.data_buffer[i]) for i in range(len(self.data_buffer))]
        indexes = [i for i in range(len(self.data_buffer))]

        combined = list(zip(indexes, mmd_score))
        sorted_combined = sorted(combined, key=lambda x: x[1])
        indexes, mmd_score = zip(*sorted_combined)
        indexes = list(indexes)

        select_buffer_idx = indexes[:self.select_buffer_num]

        select_buffer_data = select_list_by_index(self.data_buffer, select_buffer_idx)
        select_buffer_label = select_list_by_index(self.label_buffer, select_buffer_idx)
        select_buffer_weight = select_list_by_index(self.weight_buffer, select_buffer_idx)

        ori_data_temp = copy.deepcopy(self.base_data)
        ori_label_temp = copy.deepcopy(self.base_label)
        ori_weight_temp = copy.deepcopy(self.base_weight)

        delete_idx = []

        model_temp = copy.deepcopy(self.model)

        for i in range(len(select_buffer_data)):

            data_temp = select_buffer_data[i]
            label_temp = select_buffer_label[i]
            weight_temp = select_buffer_weight[i]

            retain_data = np.vstack([ori_data_temp, data_temp])
            retain_label = np.append(ori_label_temp, label_temp)
            retain_weight = np.append(ori_weight_temp, weight_temp)

            if not self.args.use_weight:
                model_temp.fit(retain_data, retain_label)
            else:
                model_temp.fit(retain_data, retain_label, sample_weight=retain_weight)

            val_acc = self.valid(model_temp)
            if val_acc >= self.original_acc:
                ori_data_temp = retain_data
                ori_label_temp = retain_label
                ori_weight_temp = retain_weight
                delete_idx.append(i)
                self.original_acc = val_acc

        del model_temp

        self.data_buffer = delete_list_by_index(self.data_buffer, delete_idx)
        self.label_buffer = delete_list_by_index(self.label_buffer, delete_idx)
        self.weight_buffer = delete_list_by_index(self.weight_buffer, delete_idx)

        if not self.args.use_weight:
            self.model = self.model.fit(ori_data_temp, ori_label_temp)
        else:
            self.model = self.model.fit(ori_data_temp, ori_label_temp, sample_weight=ori_weight_temp)
        self.base_data = ori_data_temp
        self.base_label = ori_label_temp
        self.base_weight = ori_weight_temp

        if self.args.update_bayesian:
            self.solver.update_pdf(self.model)

        return self.valid(self.model)
