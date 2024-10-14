import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import numpy as np
import copy
import math
import random
import time
from torch.utils.data import DataLoader, Dataset
from utils import MetaDataset
from torch.autograd import Variable
import tqdm


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()

        self.vars = nn.ParameterList()
        self.bn_vars = nn.ParameterList()

        # The first Linear layer
        weight = nn.Parameter(torch.ones(256, input_dim))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])

        # the first batch normal
        weight = nn.Parameter(torch.ones(256))
        bias = nn.Parameter(torch.zeros(256))
        self.vars.extend([weight, bias])

        weight = nn.Parameter(torch.zeros(256), requires_grad=False)
        bias = nn.Parameter(torch.zeros(256), requires_grad=False)
        self.bn_vars.extend([weight, bias])

        # the second linear layer
        weight = nn.Parameter(torch.ones(128, 256))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])

        # the second batch normal
        weight = nn.Parameter(torch.ones(128))
        bias = nn.Parameter(torch.zeros(128))
        self.vars.extend([weight, bias])

        weight = nn.Parameter(torch.zeros(128), requires_grad=False)
        bias = nn.Parameter(torch.zeros(128), requires_grad=False)
        self.bn_vars.extend([weight, bias])

        # the third linear layer
        weight = nn.Parameter(torch.ones(64, 128))
        nn.init.kaiming_normal_(weight)
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])

        # the third batch normal
        weight = nn.Parameter(torch.ones(64))
        bias = nn.Parameter(torch.zeros(64))
        self.vars.extend([weight, bias])

        weight = nn.Parameter(torch.zeros(64), requires_grad=False)
        bias = nn.Parameter(torch.zeros(64), requires_grad=False)
        self.bn_vars.extend([weight, bias])

    def forward(self, x, params=None, is_train=True):
        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]  # Linear 1
        enc_x1 = F.linear(x, weight=weight, bias=bias)
        weight, bias = params[2], params[3]  # BatchNorm 1
        enc_x1 = F.batch_norm(enc_x1, weight=weight, bias=bias, running_mean=self.bn_vars[0], running_var=self.bn_vars[1], training=is_train)
        enc_x1 = F.relu(enc_x1, inplace=True)

        weight, bias = params[4], params[5]
        enc_x2 = F.linear(enc_x1, weight=weight, bias=bias)
        weight, bias = params[6], params[7]
        enc_x2 = F.batch_norm(enc_x2, running_mean=self.bn_vars[2], running_var=self.bn_vars[3], weight=weight, bias=bias, training=is_train)
        enc_x2 = F.relu(enc_x2, inplace=True)

        weight, bias = params[8], params[9]
        enc_x3 = F.linear(enc_x2, weight=weight, bias=bias)
        weight, bias = params[10], params[11]
        enc_x3 = F.batch_norm(enc_x3, running_mean=self.bn_vars[4], running_var=self.bn_vars[5], weight=weight, bias=bias, training=is_train)
        enc_x3 = F.relu(enc_x3, inplace=True)

        return enc_x3

    def parameters(self):
        return self.vars


class Classifier(nn.Module):
    def __init__(self, num_cls):
        super(Classifier, self).__init__()

        self.vars = nn.ParameterList()

        # the classifier
        weight = nn.Parameter(torch.ones(num_cls, 64))
        bias = nn.Parameter(torch.zeros(num_cls))
        self.vars.extend([weight, bias])

    def forward(self, x_input, params=None):
        if params is None:
            params = self.vars

        weight, bias = params[0], params[1]
        output = F.linear(x_input, weight=weight, bias=bias)
        return output

    def parameters(self):
        return self.vars


class DNN(nn.Module):
    def __init__(self, input_dim, num_cls):
        super(DNN, self).__init__()

        self.vars = nn.ParameterList()

        self.feature_extractor = FeatureExtractor(input_dim)
        self.vars.extend(self.feature_extractor.parameters())

        self.classifier = Classifier(num_cls)
        self.vars.extend(self.classifier.parameters())

    def forward(self, x, params=None):
        if params is None:
            params = self.vars
        enc_x = self.feature_extractor(x, params[:-2])
        output = self.classifier(enc_x, params[-2:])
        return output

    def return_hidden_features(self, x):
        params = self.vars
        enc_x = self.feature_extractor(x, params[:-2])
        return enc_x

    def parameters(self):
        return self.vars


class MetaLearner(nn.Module):
    def __init__(self, input_dim, valid_db, args, device, base_data, base_label, solver, net=None):
        super(MetaLearner, self).__init__()

        if net is None:
            self.net = DNN(input_dim, args.num_cls)
        else:
            self.net = net

        self.solver = solver

        self.args = args
        self.meta_optim = optim.AdamW(self.net.parameters(), lr=args.update_lr)
        self.valid_db = valid_db
        self.device = device
        self.input_dim = input_dim
        self.meta_lr = args.meta_lr

        base_data = torch.from_numpy(base_data).to(device).float()
        base_label = torch.from_numpy(base_label).to(device).long()

        self.original_centroid = self.compute_centroid(base_data, base_label)

        self.task_buffer_data = []
        self.task_buffer_label = []
        self.task_buffer_weight = []
        self.task_buffer_data.append(base_data)
        self.task_buffer_label.append(base_label)
        self.task_buffer_weight.append(torch.from_numpy(np.array([1 for _ in range(len(base_label))])).to(device).float())
        self.task_count = 0

        self.original_valid_acc = self.valid(valid_db)
        print('Original Valid Acc: ', self.original_valid_acc)

        self.data_less_important_buffer = None

    def compute_centroid(self, base_data, base_label):
        self.net.eval()
        with torch.no_grad():
            base_feature = self.net.feature_extractor(base_data)

        base_label_one_hot = F.one_hot(base_label, self.args.num_cls).float()
        source_cen = torch.matmul(base_label_one_hot.transpose(0, 1), base_feature)
        source_centroid = (source_cen.t() / base_label_one_hot.sum(dim=0)).t()

        return source_centroid

    def validate_train_data(self, train_data, train_label):
        net = copy.deepcopy(self.net)
        optimizer = torch.optim.AdamW(net.parameters(), lr=self.args.update_lr)
        net.train()

        for _ in range(20):
            logits = net(train_data)
            loss = F.cross_entropy(logits, train_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        valid_acc = self.valid(self.valid_db, net)
        if valid_acc < self.original_valid_acc:
            del net
            return False
        else:
            del net
            return True

    def add_task_buffer(self, train_data, psuedo_train_label, label_weight):
        if self.validate_train_data(train_data, psuedo_train_label):
            if self.task_count == 0:
                self.task_buffer_data.append(train_data)
                self.task_buffer_label.append(psuedo_train_label)
                self.task_buffer_weight.append(label_weight)
            else:
                self.task_buffer_data[-1] = torch.cat([self.task_buffer_data[-1], train_data], dim=0)
                self.task_buffer_label[-1] = torch.hstack([self.task_buffer_label[-1], psuedo_train_label])
                self.task_buffer_weight[-1] = torch.hstack([self.task_buffer_weight[-1], label_weight])

            self.task_count += len(psuedo_train_label)
            if self.task_count >= self.args.final_batchsize:
                self.task_count = 0

            return True

        else:
            if self.data_less_important_buffer is None:
                self.data_less_important_buffer = train_data
            return False

    def online_update(self, train_data):
        if self.data_less_important_buffer is None:
            if self.args.use_cluster:
                psuedo_train_label, label_weight = self.return_clustering_label(train_data, True)
            else:
                psuedo_train_label, label_weight = self.return_classifier_label(train_data, True)
            self.add_task_buffer(train_data, psuedo_train_label, label_weight)

        else:
            self.data_less_important_buffer = torch.cat([self.data_less_important_buffer, train_data], dim=0)
            if self.args.use_cluster:
                psuedo_label, label_weight = self.return_clustering_label(self.data_less_important_buffer, True)
            else:
                psuedo_label, label_weight = self.return_classifier_label(self.data_less_important_buffer, True)

            data_length = len(psuedo_label) // self.args.mini_batchsize
            delete_indexes = []
            for i in range(data_length):
                data_slice, label_slice, weight_slice = self.data_less_important_buffer[i*self.args.mini_batchsize:(i+1)*self.args.mini_batchsize], \
                                          psuedo_label[i*self.args.mini_batchsize:(i+1)*self.args.mini_batchsize], \
                                          label_weight[i*self.args.mini_batchsize:(i+1)*self.args.mini_batchsize]
                flag = self.add_task_buffer(data_slice, label_slice, weight_slice)
                if flag:
                    if i != data_length-1:
                        indices_to_delete = [idx for idx in range(i*self.args.mini_batchsize, (i+1)*self.args.mini_batchsize)]
                    else:
                        indices_to_delete = [idx for idx in range(i * self.args.mini_batchsize, self.data_less_important_buffer.size(0))]
                    delete_indexes.extend(indices_to_delete)

            mask = torch.ones(self.data_less_important_buffer.size(0), dtype=torch.bool)
            mask[delete_indexes] = False
            self.data_less_important_buffer = self.data_less_important_buffer[mask]
            if self.data_less_important_buffer.size(0) == 0:
                self.data_less_important_buffer = None

        self.meta_update()
        valid_acc = self.valid(self.valid_db)

        print('Valid Accuracy: ', valid_acc)

        return valid_acc

    def normal_update(self, train_data):
        if self.args.use_cluster:
            psuedo_train_label, label_weight = self.return_clustering_label(train_data, True)
        else:
            psuedo_train_label, label_weight = self.return_classifier_label(train_data, True)

        total_loss = 0
        for _ in range(self.args.inner_step):
            logits = self.net(train_data)
            loss = F.cross_entropy(logits, psuedo_train_label)
            total_loss += loss

        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()

        valid_acc = self.valid(self.valid_db)

        print('Valid Accuracy: ', valid_acc)

        return valid_acc

    def sample_task(self):
        indexes = [i for i in range(len(self.task_buffer_label))]
        weight_list = [1/t for t in range(1, len(self.task_buffer_label)+1)]
        task_idx = random.choices(indexes, weights=weight_list, k=1)[0]
        task_data = self.task_buffer_data[task_idx]
        task_label = self.task_buffer_label[task_idx]
        task_weight = self.task_buffer_weight[task_idx]
        return task_data, task_label, task_weight, task_idx

    def sample_datapoints(self, data, label, weight):
        if data.shape[0] > self.args.mini_batchsize:
            select_indexes = random.sample([i for i in range(data.shape[0])], self.args.mini_batchsize)
            select_data = data[select_indexes, :]
            select_label = label[select_indexes]
            select_weight = weight[select_indexes]
        else:
            select_data = data
            select_label = label
            select_weight = weight

        return select_data, select_label, select_weight

    def return_classifier_label(self, data, weight=False):
        self.net.eval()
        with torch.no_grad():
            logits = self.net(data)

        psuedo_label = torch.argmax(logits, dim=-1)
        if weight:
            if self.args.use_bayesian:
                bayesian_logits = self.solver.calculate_prob(logits)
                psuedo_label = torch.argmax(bayesian_logits, dim=-1)
                weight = torch.gather(bayesian_logits, dim=1, index=psuedo_label.unsqueeze(1)).squeeze(1)
            else:
                weight = torch.gather(F.softmax(logits, dim=-1), dim=1, index=psuedo_label.unsqueeze(1)).squeeze(1)
        else:
            weight = None

        return psuedo_label, weight

    def return_clustering_label(self, data, weight=False):
        self.net.eval()
        with torch.no_grad():
            feature_enc = self.net.feature_extractor(data)

        dist = torch.cdist(feature_enc, self.original_centroid)
        psuedo_label = torch.argmin(dist, dim=-1)
        if weight:
            weight = self.generate_source_weight(feature_enc, psuedo_label)
        else:
            weight = None

        return psuedo_label, weight

    def generate_source_weight(self, source_features, source_label):
        source_cos_sim = source_features.unsqueeze(1) * self.original_centroid.unsqueeze(0)
        x = source_cos_sim.sum(2) / (source_features.norm(2, dim=1, keepdim=True) * self.original_centroid.norm(2, dim=1, keepdim=True).t() + 1e-4)
        source_sim_norm = 0.5 * (1 + x)
        source_weight = torch.gather(source_sim_norm, dim=1, index=source_label.unsqueeze(1)).squeeze(1)
        return source_weight

    def meta_update(self):
        self.net.train()
        if self.args.use_meta_online:
            total_loss = 0
            for _ in range(self.args.N_update_test):
                task_data, task_label, task_weight, task_id = self.sample_task()
                tr_x, tr_y, tr_w = self.sample_datapoints(task_data, task_label, task_weight)

                spts_x, spts_y, spts_w, qry_x, qry_y, qry_w = tr_x[:self.args.support, :], tr_y[:self.args.support], \
                                                              tr_w[:self.args.support], tr_x[self.args.support:, :], \
                                                              tr_y[self.args.support:], tr_w[self.args.support:]
                logits = self.net(spts_x)
                if self.args.use_weight:
                    loss = self.calculate_classify_loss(logits, spts_y, spts_w)
                else:
                    loss = F.cross_entropy(logits, spts_y)
                grad = torch.autograd.grad(loss, self.net.parameters())
                fast_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, self.net.parameters())))
                for _ in range(1, self.args.inner_step):
                    logits = self.net(spts_x, fast_weights)
                    if self.args.use_weight:
                        loss = self.calculate_classify_loss(logits, spts_y, spts_w)
                    else:
                        loss = F.cross_entropy(logits, spts_y)
                    grad = torch.autograd.grad(loss, fast_weights)
                    fast_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, fast_weights)))

                query_logits = self.net(qry_x, fast_weights)
                if self.args.use_weight:
                    query_loss = self.calculate_classify_loss(query_logits, qry_y, qry_w)
                else:
                    query_loss = F.cross_entropy(query_logits, qry_y)
                total_loss += query_loss

            self.meta_optim.zero_grad()
            total_loss.backward()
            self.meta_optim.step()

        else:
            for _ in range(self.args.N_update_test):
                task_data, task_label, task_weight, task_id = self.sample_task()
                tr_x, tr_y, tr_w = self.sample_datapoints(task_data, task_label, task_weight)

                query_logits = self.net(tr_x)
                if self.args.use_weight:
                    query_loss = self.calculate_classify_loss(query_logits, tr_y, tr_w)
                else:
                    query_loss = F.cross_entropy(query_logits, tr_y)

                self.meta_optim.zero_grad()
                query_loss.backward()
                self.meta_optim.step()

        if self.args.use_bayesian:
            self.solver.update(self.net)

        if self.args.use_cluster:
            self.update_centroid()

    def update_centroid(self):
        self.net.eval()
        buffer_features = None
        buffer_feature_list = []
        buffer_labels = None
        with torch.no_grad():
            for i in range(len(self.task_buffer_label)):
                temp_buffer_data = self.task_buffer_data[i]
                temp_buffer_label = self.task_buffer_label[i]
                temp_buffer_weight = self.task_buffer_weight[i]
                temp_feature = self.net.feature_extractor(temp_buffer_data)
                temp_feature = temp_feature*temp_buffer_weight.unsqueeze(1)
                if i == 0:
                    buffer_features = temp_feature
                    buffer_labels = temp_buffer_label
                else:
                    buffer_features = torch.cat([buffer_features, temp_feature], dim=0)
                    buffer_labels = torch.hstack([buffer_labels, temp_buffer_label])

                    buffer_feature_list.append(temp_feature)

        base_label_one_hot = F.one_hot(buffer_labels, self.args.num_cls).float()
        source_cen = torch.matmul(base_label_one_hot.transpose(0, 1), buffer_features)
        self.original_centroid = (source_cen.t() / base_label_one_hot.sum(dim=0)).t()

        self.original_centroid = self.kmeans(buffer_features, self.original_centroid, self.args.num_cls)

        for i in range(1, len(self.task_buffer_label)):
            # self.task_buffer_label[i] = torch.argmin(torch.cdist(buffer_feature_list[i-1], self.original_centroid), dim=-1)
            self.task_buffer_weight[i] = self.generate_source_weight(buffer_feature_list[i - 1], self.task_buffer_label[i])

    def kmeans(self, all_fea, cen, num_classes):
        dist = torch.cdist(all_fea, cen, p=2)
        pred = torch.argmin(dist, dim=1)

        for _ in range(5):
            all_output = F.one_hot(pred, num_classes).float()
            cen = torch.matmul(all_output.transpose(0, 1), all_fea)
            cen = (cen.t() / (all_output.sum(dim=0)+1e-4)).t()
            dist = torch.cdist(all_fea, cen, p=2)
            pred = torch.argmin(dist, dim=1)

        target_centroid = cen

        return target_centroid

    def valid(self, valid_db, net=None):
        if net is None:
            net = self.net
        net.eval()
        acc_list = []
        for v_data, v_label in valid_db:
            v_data = v_data.to(self.device).float()
            v_label = v_label.to(self.device).long()
            logits = net(v_data)
            pred = torch.argmax(logits, dim=-1)
            acc = torch.eq(pred, v_label).sum() / len(v_label)
            acc_list.append(acc.detach().cpu().numpy())

        return np.mean(acc_list)

    def calculate_classify_loss(self, output, target, source_weight):
        prob_p = F.softmax(output, dim=-1)

        prob_q = Variable(torch.cuda.FloatTensor(prob_p.size()).fill_(0))
        prob_q.scatter_(1, target.unsqueeze(1), torch.ones(prob_p.size(0), 1).cuda())

        loss = - (source_weight * (prob_q * F.log_softmax(output, dim=1)).sum(1)).mean()

        return loss

    def predict(self, x):
        with torch.no_grad():
            output = self.net(x)

        output_pred = F.softmax(output, dim=-1)
        pred_label = torch.argmax(output_pred, dim=-1)

        return pred_label, output_pred


class Pretrain_Meta_Learner(nn.Module):
    def __init__(self, args, device, data, label, input_dim, num_cls):
        super(Pretrain_Meta_Learner, self).__init__()

        self.args = args
        self.support = args.support
        self.query = args.query
        self.device = device
        self.input_dim = input_dim

        self.data = torch.from_numpy(data).to(device).float()
        self.label = torch.from_numpy(label).to(device).long()

        self.task_buffer = self.construct_multi_task()

        self.net = DNN(input_dim, num_cls)

        self.update_lr = args.pretrain_update_lr
        self.meta_lr = args.pretrain_meta_lr

        self.meta_optim = torch.optim.AdamW(self.net.parameters(), lr=self.update_lr)

    def construct_multi_task(self):
        total_data, total_label = None, None
        indexes = np.array([i for i in range(len(self.label))])
        for i in range(3):
            np.random.shuffle(indexes)
            data, label = self.data[indexes, :], self.label[indexes]
            if i == 0:
                total_data, total_label = data, label
            else:
                total_data = torch.cat((total_data, data), dim=0)
                total_label = torch.hstack((total_label, label))

        total_set = MetaDataset(total_data, total_label)
        total_db = DataLoader(total_set, batch_size=self.args.support+self.args.query, shuffle=True, drop_last=True)

        return total_db

    def meta_train(self):
        self.net.train()
        total_loss = 0
        for data, label in self.task_buffer:
            spts_x, spts_label = data[:self.args.support, :], label[:self.args.support]
            qry_data, qry_label = data[self.args.support:, :], label[self.args.support:]

            logits = self.net(spts_x, params=None)
            loss = F.cross_entropy(logits, spts_label)
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, self.net.parameters())))

            for _ in range(1, self.args.inner_step):
                logits = self.net(spts_x, fast_weights)
                loss = F.cross_entropy(logits, spts_label)
                grad = torch.autograd.grad(loss, fast_weights)
                fast_weights = list(map(lambda p: p[1] - self.meta_lr * p[0], zip(grad, fast_weights)))

            logits_q = self.net(qry_data, fast_weights)
            loss_q = F.cross_entropy(logits_q, qry_label)
            total_loss += loss_q

        self.meta_optim.zero_grad()
        total_loss.backward()
        self.meta_optim.step()

    def valid_data(self, valid_db):
        acc_list = []
        self.net.eval()
        for v_data, v_label in valid_db:
            v_data = v_data.to(self.device).float()
            v_label = v_label.to(self.device).long()
            pred = torch.argmax(self.net(v_data), dim=-1)
            acc = torch.eq(pred, v_label).sum() / len(v_label)
            acc_list.append(acc.detach().cpu().data)
        return np.mean(acc_list)

    def return_weight(self):
        return self.net.parameters()


class OriginalPretain(nn.Module):
    def __init__(self, input_dim, args, base_data, base_label, device):
        super(OriginalPretain, self).__init__()

        self.device = device
        self.net = DNN(input_dim, args.num_cls)
        self.update_lr = args.update_lr
        self.optim = optim.AdamW(self.net.parameters(), lr=self.update_lr)

        self.data_loader = DataLoader(MetaDataset(base_data, base_label), batch_size=args.mini_batchsize, shuffle=True,
                                      drop_last=True)

    def meta_train(self):
        self.net.train()
        for data, label in self.data_loader:
            data = data.to(self.device).float()
            label = label.to(self.device).long()

            logits = self.net(data)
            loss = F.cross_entropy(logits, label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def valid_data(self, valid_db):
        acc_list = []
        self.net.eval()
        for v_data, v_label in valid_db:
            v_data = v_data.to(self.device).float()
            v_label = v_label.to(self.device).long()
            pred = torch.argmax(self.net(v_data), dim=-1)
            acc = torch.eq(pred, v_label).sum() / len(v_label)
            acc_list.append(acc.detach().cpu().data)
        return np.mean(acc_list)
