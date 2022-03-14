#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: model.py
# Time: 2021/05
# Author: Zhu Yutao
# Description:
from unittest import TestLoader
from xml.sax.xmlreader import InputSource
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np
import os
import pandas as pd
import program_logging
import progressbar as bar
# import tensorflow as tf
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import *
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(1)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class TestLinear(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(TestLinear, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = self.fc1(x)
        out = self.fc2(x)
        return out


class TestLstm(nn.Module):
    def __init__(self, input_size, hid_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hid_size = hid_size
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hid_size, batch_first=True)
        self.fc = nn.Linear(self.hid_size, 2)

    def forward(self, x):
        x = x.view(self.batch_size, -1, self.input_size)
        x, _ = self.lstm(x)
        x = x.view(x.shape[0] * x.shape[1], self.hid_size)
        out = self.fc(x)
        return out
        

class MyNet(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        out = self.fc2(x)
        return out


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, num_layers, batch_size, output_size, batch_first=False):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            bidirectional=False)
        self.hidden2label = nn.Linear(in_features=hidden_dim,
                                      out_features=output_size)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_dim).to(device)
        return h_0, c_0

    def forward(self, data):
        self.hidden = self.init_hidden()
        result, self.hidden = self.lstm(data, self.hidden)
        result = result.contiguous().view(data.shape[0] * data.shape[1], self.hidden_dim)
        result = self.hidden2label(result)
        return result


def loss_func(result, label, loss_type, loss_weight):
    if loss_type == 'CrossEntropyLoss':
        loss_ce = nn.CrossEntropyLoss(weight=loss_weight)
        loss = loss_ce(result, label)
    elif loss_type == 'BCEWithLogitsLoss':
        loss_bce = nn.BCEWithLogitsLoss()
        label_one_hot = F.one_hot(label, num_classes=2).float()
        loss = loss_bce(result, label_one_hot)
    else:
        logging.error('ERROR: Invalid loss function!')
        return
    return loss


def statistic(result: torch.Tensor, label: torch.Tensor):
    tp = ((label == 1) & (result == 1)).sum().float().item()
    tn = ((label == 0) & (result == 0)).sum().float().item()
    fp = ((label == 0) & (result == 1)).sum().float().item()
    fn = ((label == 1) & (result == 0)).sum().float().item()
    return tp, tn, fp, fn


def evaluate(tfpn: list):
    tp = tfpn[0]
    tn = tfpn[1]
    fp = tfpn[2]
    fn = tfpn[3]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
    else:
        precision = 0
        recall = 0
        f1 = 0
    tpr = recall
    fpr = fp / (fp + tn)
    return accuracy, precision, recall, f1, tpr, fpr


def data_preprocess(filepath: str, index_list: list, time_step: int, batch_size=1, delta=0):
    """
    数据预处理

    :param filepath: 数据文件路径（包括特征及标签）
    :param index_list:
    :param time_step: 序列片段长度
    :param delta: 取片段偏移
    :return:
    """
    file_df = pd.read_csv(filepath_or_buffer=filepath)
    data_df = file_df.loc[:, index_list]
    data_df = (data_df - data_df.min()) / (data_df.max() - data_df.min())
    data = data_df.values
    label = file_df['Label'].values

    len_data = data.shape[0]
    # print(len_data)
    # width_data = int(data.size / len_data)
    # num = time_step * batch_size
    batch_len = batch_size * (time_step - delta) + delta
    batch_num = (len_data - delta) // (batch_len - delta)
    data_list = []
    label_list = []
    for i in range(batch_num):
        batch_data = []
        batch_label = []
        for j in range(batch_size):
            start = i * (batch_len - delta) + j * (time_step - delta)
            end = i * (batch_len - delta) + j * (time_step - delta) + time_step
            batch_data.extend(data[start:end])
            batch_label.extend(label[start:end])
        data_list.append(batch_data)
        label_list.append(batch_label)
    data_pre = np.array(data_list)
    label_pre = np.array(label_list)

    return data_pre, label_pre


def train():
    # 文件准备 ######################################################################
    time_now = time.strftime('%Y%m%d%H%M%S', time.localtime())
    model_name = 'LSTM'     # LSTM, MyNet
    dir_result = 'train/17_ts_pkt/' + model_name + '/' + time_now
    os.makedirs(name=dir_result, exist_ok=True)
    dir_tensorboard = os.path.join(dir_result, 'runs')
    path_log = os.path.join(dir_result, 'log')
    path_loss_train = os.path.join(dir_result, 'loss_train')
    path_eva_train = os.path.join(dir_result, 'evaluation_train')
    path_loss_test = os.path.join(dir_result, 'loss_test')
    path_eva_test = os.path.join(dir_result, 'evaluation_test')

    writer = SummaryWriter(dir_tensorboard)
    file_log = open(path_log, 'w')
    file_loss_train = open(path_loss_train, 'w')
    file_eva_train = open(path_eva_train, 'w')
    file_loss_test = open(path_loss_test, 'w')
    file_eva_test = open(path_eva_test, 'w')

    logging.info('Device: ' + str(device))

    file_log.write('Time: ' + time_now + '\n')

    # 数据预处理 ######################################################################
    logging.info('Data preprocessing ...')
    time_step = 1000
    batch_size = 8
    delta = time_step - 1000
    train_filepath = 'data/data_video/vall/index_ts_label1_v1_v10_0&1_1&2'
    test_filepath = 'data/data_video/vall/index_ts_label0_v11_v13_0&1_1&2'
    index_list = pd.read_csv(filepath_or_buffer='data/data_video/vall/mrmr/feature_selection.txt',
                             header=None).values.tolist()[0]
    train_data, train_label = data_preprocess(filepath=train_filepath,
                                              index_list=index_list,
                                              time_step=time_step,
                                              batch_size=batch_size,
                                              delta=delta)
    test_data, test_label = data_preprocess(filepath=test_filepath,
                                            index_list=index_list,
                                            time_step=time_step,
                                            batch_size=batch_size)
    num_feature = train_data.shape[2]
    num_pos = sum(sum(train_label == 1))
    num_neg = sum(sum(train_label == 0))
    w_pos = num_neg / (num_pos + num_neg)
    w_neg = num_pos / (num_pos + num_neg)

    file_log.write('****** Data Preprocess ******\n')
    file_log.write('time_step=' + str(time_step) + '\n')
    file_log.write('batch_size=' + str(batch_size) + '\n')
    file_log.write('delta=' + str(delta) + '\n')
    file_log.write('****** Dataset ******\n')
    file_log.write('num_feature=' + str(num_feature) + '\n')
    file_log.write('[Train Data]\n')
    file_log.write(train_filepath + '\n')
    file_log.write('[Test Data]\n')
    file_log.write(test_filepath + '\n')

    # 模型构建 ########################################################################
    logging.info('Model creating ...')
    hidden_dim = 64
    num_layers = 1
    learning_rate = 1e-3
    weight_decay = 1e-2
    loss_type = 'CrossEntropyLoss'     # CrossEntropyLoss, BCEWithLogitsLoss
    loss_weight = torch.tensor(data=[w_neg, w_pos], dtype=torch.float32).to(device)
    epochs = 1000
    if model_name == 'LSTM':
        model = LSTM(input_size=num_feature,
                     hidden_dim=hidden_dim,
                     num_layers=num_layers,
                     batch_size=batch_size,
                     output_size=2).to(device)
    else:
        model = MyNet(input_size=num_feature,
                      hidden_dim=hidden_dim,
                      output_size=2).to(device)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    file_log.write('****** Model ******\n')
    file_log.write('[Parameters]\n')
    file_log.write('model=' + str(type(model)) + '\n')
    file_log.write('hidden_dim=' + str(hidden_dim) + '\n')
    file_log.write('num_layers=' + str(num_layers) + '\n')
    file_log.write('loss_type=' + str(loss_type) + '\n')
    file_log.write('loss_weight=' + str(loss_weight.tolist()) + '\n')
    file_log.write('optimizer=' + str(type(optimizer)) + '\n')
    file_log.write('learning_rate=' + str(learning_rate) + '\n')
    file_log.write('weight_decay=' + str(weight_decay) + '\n')
    file_log.write('[Structure]\n')
    for params in model.state_dict():
        file_log.write("{}\t{}\n".format(params, model.state_dict()[params]))

    # 训练前检验 #####################################################################################################
    logging.info('Test before train.')
    result_test = []
    tfpn_test = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            if model_name == 'LSTM':
                data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1).to(device)
            else:
                data_tensor = torch.tensor(data=data, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(data=test_label[idx], dtype=torch.long).to(device)
            res = model(data_tensor)
            res = res.argmax(dim=1)
            result_test.extend(res.tolist())
            tfpn = statistic(res, label_tensor)
            tfpn_test = (np.array(tfpn_test) + np.array(tfpn)).tolist()
    result_series = pd.Series(result_test)
    result_series.to_csv(path_or_buf=os.path.join(dir_result, 'test_label'),
                         header=False,
                         index=False)
    eva_test = evaluate(tfpn_test)
    logging.info('Test accuracy=%f, precision=%f, recall=%f, f1=%f, tpr=%f, fpr=%f',
                 eva_test[0], eva_test[1], eva_test[2], eva_test[3], eva_test[4], eva_test[5])

    file_log.write('****** Test Before Train ******\n')
    file_log.write('accuracy=' + str(eva_test[0]) + '\n')
    file_log.write('precision=' + str(eva_test[1]) + '\n')
    file_log.write('recall=' + str(eva_test[2]) + '\n')
    file_log.write('f1=' + str(eva_test[3]) + '\n')
    file_log.write('tpr=' + str(eva_test[4]) + '\n')
    file_log.write('fpr=' + str(eva_test[5]) + '\n')

    # 训练 #########################################################################################################
    logging.info('Training ...')
    file_loss_train.write('loss,tp,tn,fp,fn\n')
    file_eva_train.write('loss,accuracy,precision,recall,f1,tpr,fpr\n')
    file_loss_test.write('loss,tp,tn,fp,fn\n')
    file_eva_test.write('loss,accuracy,precision,recall,f1,tpr,fpr\n')
    for epoch in range(epochs):
        # 训练集训练
        loss_train = 0.0
        tfpn_train = [0.0, 0.0, 0.0, 0.0]
        len_train = train_data.shape[0]
        bar_train = bar.ProgressBar()
        for idx, data in enumerate(train_data):
            optimizer.zero_grad()
            if model_name == 'LSTM':
                data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1).to(device)
            else:
                data_tensor = torch.tensor(data=data, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(data=train_label[idx], dtype=torch.long).to(device)
            res = model(data_tensor)
            loss = loss_func(result=res, label=label_tensor, loss_type=loss_type, loss_weight=loss_weight)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            res = res.argmax(dim=1)
            tfpn = statistic(res, label_tensor)
            tfpn_train = (np.array(tfpn_train) + np.array(tfpn)).tolist()
            file_loss_train.write(str(loss.item()) + ',' +
                            str(tfpn[0]) + ',' +
                            str(tfpn[1]) + ',' +
                            str(tfpn[2]) + ',' +
                            str(tfpn[3]) + '\n')
            bar_train.update(idx)
        # 测试集测试
        loss_test = 0.0
        tfpn_test = [0.0, 0.0, 0.0, 0.0]
        len_test = test_data.shape[0]
        for idx, data in enumerate(test_data):
            if model_name == 'LSTM':
                data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1).to(device)
            else:
                data_tensor = torch.tensor(data=data, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(data=test_label[idx], dtype=torch.long).to(device)
            res = model(data_tensor)
            loss = loss_func(result=res, label=label_tensor, loss_type=loss_type, loss_weight=loss_weight)
            loss_test += loss.item()
            res = res.argmax(dim=1)
            result_test.extend(res.tolist())
            tfpn = statistic(res, label_tensor)
            tfpn_test = (np.array(tfpn_test) + np.array(tfpn)).tolist()
            file_loss_test.write(str(loss.item()) + ',' +
                                 str(tfpn[0]) + ',' +
                                 str(tfpn[1]) + ',' +
                                 str(tfpn[2]) + ',' +
                                 str(tfpn[3]) + '\n')
        # 结果统计
        eva_train = evaluate(tfpn_train)
        loss_train_avg = loss_train / len_train
        writer.add_scalar('training loss', loss_train_avg, epoch)
        file_eva_train.write(str(loss_train_avg) + ',' +
                             str(eva_train[0]) + ',' +
                             str(eva_train[1]) + ',' +
                             str(eva_train[2]) + ',' +
                             str(eva_train[3]) + ',' +
                             str(eva_train[4]) + ',' +
                             str(eva_train[5]) + '\n')
        eva_test = evaluate(tfpn_test)
        loss_test_avg = loss_test / len_test
        writer.add_scalar('test loss', loss_test_avg, epoch)
        file_eva_test.write(str(loss_test_avg) + ',' +
                            str(eva_test[0]) + ',' +
                            str(eva_test[1]) + ',' +
                            str(eva_test[2]) + ',' +
                            str(eva_test[3]) + ',' +
                            str(eva_test[4]) + ',' +
                            str(eva_test[5]) + '\n')
        logging.info('\nEpoch(%d/%d)' +
                     '\n - [Train] loss=%f, accuracy=%f, precision=%f, recall=%f, f1=%f, tpr=%f, fpr=%f' +
                     '\n - [Test]  loss=%f, accuracy=%f, precision=%f, recall=%f, f1=%f, tpr=%f, fpr=%f',
                     epoch + 1, epochs,
                     loss_train_avg, eva_train[0], eva_train[1], eva_train[2], eva_train[3], eva_train[4], eva_train[5],
                     loss_test_avg, eva_test[0], eva_test[1], eva_test[2], eva_test[3], eva_test[4], eva_test[5])
    torch.save(obj=model, f=os.path.join(dir_result, 'model.pkl'))  # 保存模型

    file_log.write('****** Train ******\n')
    file_log.write('epochs=' + str(epochs) + '\n')

    # 训练后结果 ######################################################################################################
    result_test = []
    tfpn_test = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for idx, data in enumerate(test_data):
            if model_name == 'LSTM':
                data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1).to(device)
            else:
                data_tensor = torch.tensor(data=data, dtype=torch.float32).to(device)
            label_tensor = torch.tensor(data=test_label[idx], dtype=torch.long).to(device)
            res = model(data_tensor)
            res = res.argmax(dim=1)
            result_test.extend(res.tolist())
            tfpn = statistic(res, label_tensor)
            tfpn_test = (np.array(tfpn_test) + np.array(tfpn)).tolist()
    result_series = pd.Series(result_test)
    result_series.to_csv(path_or_buf=os.path.join(dir_result, 'result_label'),
                         header=False,
                         index=False)
    eva_test = evaluate(tfpn_test)
    logging.info('Result accuracy=%f, precision=%f, recall=%f, f1=%f, tpr=%f, fpr=%f',
                 eva_test[0], eva_test[1], eva_test[2], eva_test[3], eva_test[4], eva_test[5])

    file_log.write('****** Test After Train ******\n')
    file_log.write('accuracy=' + str(eva_test[0]) + '\n')
    file_log.write('precision=' + str(eva_test[1]) + '\n')
    file_log.write('recall=' + str(eva_test[2]) + '\n')
    file_log.write('f1=' + str(eva_test[3]) + '\n')
    file_log.write('tpr=' + str(eva_test[4]) + '\n')
    file_log.write('fpr=' + str(eva_test[5]) + '\n')

    # 结束 ##########################################################################################################
    writer.close()
    file_log.close()
    file_loss_train.close()
    file_eva_train.close()
    file_loss_test.close()
    file_eva_test.close()


def get_test_dataloader():
    """
    测试数据

    :return:
    """
    NUM_DATA = 10000
    BATCH_SIZE = 100
    NUM_FEATURE = 10
    np.random.seed(1)
    train_data = torch.FloatTensor(np.random.random([NUM_DATA, NUM_FEATURE]))
    train_label = torch.from_numpy(np.array([int(sum(x)/(0.5*NUM_FEATURE)) for x in train_data]))
    test_data = torch.FloatTensor(np.random.random([NUM_DATA, NUM_FEATURE]))
    test_label = torch.from_numpy(np.array([int(sum(x)/(0.5*NUM_FEATURE)) for x in test_data]))
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
    return (train_dataloader, test_dataloader), NUM_FEATURE


def get_real_dataloader(batch_size: int):
    """
    实际数据

    :return:
    """
    # BATCH_SIZE = 100
    index_list = pd.read_csv(filepath_or_buffer='data/data_video/vall/mrmr/feature_selection.txt',
                             header=None).values.tolist()[0]
    # 训练集
    filepath = 'data/data_video/vall/index_ts_label1_v1_v10_0&1_1&2'
    file_df = pd.read_csv(filepath_or_buffer=filepath)
    data_df = file_df.loc[:, index_list]
    num_feature = data_df.shape[1]
    data_avg = data_df.mean()
    data_std = data_df.std()
    data_df = (data_df - data_avg) / data_std
    data = data_df.values
    data = torch.FloatTensor(data)
    label = torch.from_numpy(file_df['Label'].values)
    dataset = TensorDataset(data, label)
    sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    # 测试集
    filepath = 'data/data_video/vall/index_ts_label0_v11_v13_0&1_1&2'
    file_df = pd.read_csv(filepath_or_buffer=filepath)
    data_df = file_df.loc[:, index_list]
    data_df = (data_df - data_avg) / data_std
    data = torch.FloatTensor(data_df.values)
    label = torch.from_numpy(file_df['Label'].values)
    dataset = TensorDataset(data, label)
    sampler = SequentialSampler(dataset)
    test_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    return (train_dataloader, test_dataloader), num_feature


def test_model():
    # 参数设置
    BATCH_SIZE = 100
    HIDDEN_SIZE = 128
    NUM_LAYERS = 1

    # 文件准备
    time_now = time.strftime('%Y%m%d%H%M%S', time.localtime())
    dir_root = 'train'
    dir_train = os.path.join(dir_root, 'test')
    if not os.path.exists(dir_train):
        os.mkdir(dir_train)
    dir_result = os.path.join(dir_train, time_now)
    if not os.path.exists(dir_result):
        os.mkdir(dir_result)
    dir_tensorboard = os.path.join(dir_result, 'runs')
    writer = SummaryWriter(dir_tensorboard)

    # 测试数据
    (train_dataloader, test_dataloader), NUM_FEATURE = get_real_dataloader(BATCH_SIZE)

    # 测试模型
    # model = TestLinear(NUM_FEATURE, HIDDEN_SIZE, 2).to(device)
    model = TestLstm(NUM_FEATURE, HIDDEN_SIZE, BATCH_SIZE).to(device)
    # model = MyNet(NUM_FEATURE, HIDDEN_SIZE, 2).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters())

    # 测试训练
    best_acc = 0.0
    for epoch in range(1, 101):
        # train
        model.train()
        criterion = nn.CrossEntropyLoss()
        loss_sum = 0.0
        num = 0.0
        for batch_idx, (x, y) in enumerate(train_dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            num += 1
        loss_train = loss_sum / num
        print('Train Epoch: {}\tloss={:.6f}'.format(epoch, loss_train))
        # test
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss_test = 0.0
        acc = 0
        for batch_idx, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                y_ = model(x)
            loss_test += criterion(y_, y)
            pred = y_.max(-1, keepdim=True)[1]
            acc += pred.eq(y.view_as(pred)).sum().item()
        loss_test /= len(test_dataloader.dataset)
        print('\nTest set: Average loss={:.4f}, Accuracy={}/{} ({:.0f}%)'.format(
            loss_test, acc, len(test_dataloader.dataset),
            100. * acc / len(test_dataloader.dataset)))
        acc = acc / len(test_dataloader.dataset)
        if best_acc < acc:
            best_acc = acc
        print("acc={:.4f}, best_acc={:.4f}\n".format(acc, best_acc))
        writer.add_scalar('training loss', loss_train, epoch)
        writer.add_scalar('test loss', loss_test, epoch)

    writer.close()


if __name__ == '__main__':
    train()
