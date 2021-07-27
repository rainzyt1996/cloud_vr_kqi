#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Software: PyCharm
# File: model.py
# Time: 2021/05
# Author: Zhu Yutao
# Description:
import logging
import numpy as np
import os
import pandas as pd
import program_logging
import progressbar as bar
import tensorflow as tf
import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)


class BiLSTM(nn.Module):

    def __init__(self, input_size, hidden_dim, batch_size, output_size, loss_weight):
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)
        self.hidden2label = nn.Linear(in_features=hidden_dim,
                                      out_features=output_size)

        self.loss = nn.CrossEntropyLoss(weight=loss_weight)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        h_0 = torch.randn(2, self.batch_size, self.hidden_dim // 2)
        c_0 = torch.randn(2, self.batch_size, self.hidden_dim // 2)
        return h_0, c_0

    def _get_lstm_features(self, data):
        self.hidden = self.init_hidden()
        lstm_out, self.hidden = self.lstm(data, self.hidden)
        lstm_out = lstm_out.view(data.shape[0] * data.shape[1], self.hidden_dim)
        lstm_feats = self.hidden2label(lstm_out)
        return lstm_feats

    def bce_loss(self, data, label):
        feats = self._get_lstm_features(data)
        loss = self.loss(feats, label)
        res = feats.argmax(dim=1)
        tfpn = statistic(res, label)
        return loss, tfpn

    def forward(self, data):
        lstm_feats = self._get_lstm_features(data)
        result = lstm_feats.argmax(dim=1)
        return result


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
    if tp + fp:
        precision = tp / (tp + fp)
    else:
        precision = 2
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    return accuracy, precision, recall, f1


def data_preprocess(filepath: str, time_step: int, batch_size=1, delta=0):
    """
    数据预处理

    :param filepath: 数据文件路径（包括特征及标签）
    :param batch_size:
    :param time_step: 序列片段长度
    :param delta: 取片段偏移
    :return:
    """
    file_df = pd.read_csv(filepath_or_buffer=filepath)
    data_df = file_df.loc[:, ('T1_T0_Sync', 'T2_T1_Sync')]
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
    str_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    result_dir = 'data/data_video/result/result_' + str_time
    os.mkdir(result_dir)

    hidden_dim = 128
    time_step = 100
    batch_size = 4
    delta = 90
    learning_rate = 0.01
    weight_decay = 1e-4
    loss_weight = torch.tensor(data=[0.1, 0.9], dtype=torch.float32)

    log_list = []
    log_list.append(['Time: ' + str_time])
    log_list.append(['****** Parameter ******'])
    log_list.append(['hidden_dim=' + str(hidden_dim)])
    log_list.append(['time_step=' + str(time_step)])
    log_list.append(['batch_size=' + str(batch_size)])
    log_list.append(['delta=' + str(delta)])
    log_list.append(['learning_rate=' + str(learning_rate)])
    log_list.append(['weight_decay=' + str(weight_decay)])
    log_list.append(['loss_weight=' + str(loss_weight.tolist())])

    # 数据预处理
    logging.info('Data preprocessing ...')
    train_filepath_1 = 'data/data_video/data_video_1/log/index_label'
    train_filepath_2 = 'data/data_video/data_video_2/log/index_label'
    test_filepath = 'data/data_video/data_video_3/log/index_label'
    train_data_1, train_label_1 = data_preprocess(filepath=train_filepath_1,
                                                  time_step=time_step,
                                                  batch_size=batch_size,
                                                  delta=delta)
    train_data_2, train_label_2 = data_preprocess(filepath=train_filepath_2,
                                                  time_step=time_step,
                                                  batch_size=batch_size,
                                                  delta=delta)
    train_data = np.concatenate((train_data_1, train_data_2), axis=0)
    train_label = np.concatenate((train_label_1, train_label_2), axis=0)
    test_data, test_label = data_preprocess(filepath=test_filepath,
                                            time_step=time_step,
                                            batch_size=batch_size)
    num_feature = train_data.shape[2]

    log_list.append(['num_feature=' + str(num_feature)])
    log_list.append(['****** Dataset ******'])
    log_list.append(['Train Data:'])
    log_list.append([train_filepath_1])
    log_list.append([train_filepath_2])
    log_list.append(['Test Data:'])
    log_list.append([test_filepath])

    # 模型构建
    logging.info('Model creating ...')
    model = BiLSTM(input_size=num_feature,
                   hidden_dim=hidden_dim,
                   batch_size=batch_size,
                   output_size=2,
                   loss_weight=loss_weight)
    optimizer = optim.Adam(params=model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)

    log_list.append(['****** Model ******'])
    log_list.append(['model=BiLSTM'])
    log_list.append(['optimizer=Adam'])

    # 检验
    logging.info('Test before train.')
    result = []
    tfpn_sum = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for index, data in enumerate(test_data):
            data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1)
            label_tensor = torch.tensor(data=test_label[index], dtype=torch.long)
            res = model(data_tensor)
            result.extend(res.tolist())
            tfpn = statistic(res, label_tensor)
            tfpn_sum = (np.array(tfpn_sum) + np.array(tfpn)).tolist()
    result_series = pd.Series(result)
    result_series.to_csv(path_or_buf=os.path.join(result_dir, 'test_label'),
                         header=False,
                         index=False)
    accuracy, precision, recall, f1 = evaluate(tfpn_sum)
    logging.info('Test accuracy=%f, precision=%f, recall=%f, f1=%f', accuracy, precision, recall, f1)

    log_list.append(['****** Test Before Train ******'])
    log_list.append(['accuracy=' + str(accuracy)])
    log_list.append(['precision=' + str(precision)])
    log_list.append(['recall=' + str(recall)])
    log_list.append(['f1=' + str(f1)])

    # 训练
    logging.info('Training ...')
    epochs = 10
    loss_list = []
    loss_acc_list = []
    for epoch in range(epochs):
        loss_sum = 0.
        tfpn_sum = [0.0, 0.0, 0.0, 0.0]
        len_data = train_data.shape[0]
        bar_train = bar.ProgressBar()
        for index, data in enumerate(train_data):
            optimizer.zero_grad()
            data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1)
            label_tensor = torch.tensor(data=train_label[index], dtype=torch.long)
            loss, tfpn = model.bce_loss(data=data_tensor, label=label_tensor)
            loss_list.append([loss.item(), tfpn])
            loss_sum += loss.item()
            tfpn_sum = (np.array(tfpn_sum) + np.array(tfpn)).tolist()
            loss.backward()
            optimizer.step()
            bar_train.update(index)
        accuracy, precision, recall, f1 = evaluate(tfpn_sum)
        loss_acc_list.append([loss_sum / len_data, accuracy, precision, recall, f1])
        logging.info('\nEpoch(%d/%d): loss=%f, accuracy=%f, precision=%f, recall=%f, f1=%f',
                     epoch + 1, epochs, loss_sum / len_data, accuracy, precision, recall, f1)
    loss_df = pd.DataFrame(data=loss_list)
    loss_df.to_csv(path_or_buf=os.path.join(result_dir, 'loss_list'), index=False)
    loss_acc_df = pd.DataFrame(data=loss_acc_list, columns=['loss', 'accuracy', 'precision', 'recall', 'f1'])
    loss_acc_df.to_csv(path_or_buf=os.path.join(result_dir, 'loss_acc_list'), index=False)
    torch.save(obj=model, f=os.path.join(result_dir, 'model.pkl'))  # 保存模型

    log_list.append(['****** Train ******'])
    log_list.append(['epochs=' + str(epochs)])

    # 结果
    result = []
    tfpn_sum = [0.0, 0.0, 0.0, 0.0]
    with torch.no_grad():
        for index, data in enumerate(test_data):
            data_tensor = torch.tensor(data=data, dtype=torch.float32).view(time_step, batch_size, -1)
            label_tensor = torch.tensor(data=test_label[index], dtype=torch.long)
            res = model(data_tensor)
            result.extend(res.tolist())
            tfpn = statistic(res, label_tensor)
            tfpn_sum = (np.array(tfpn_sum) + np.array(tfpn)).tolist()
    result_series = pd.Series(result)
    result_series.to_csv(path_or_buf=os.path.join(result_dir, 'result_label'),
                         header=False,
                         index=False)
    accuracy, precision, recall, f1 = evaluate(tfpn_sum)
    logging.info('Test accuracy=%f, precision=%f, recall=%f, f1=%f', accuracy, precision, recall, f1)

    log_list.append(['****** Test After Train ******'])
    log_list.append(['accuracy=' + str(accuracy)])
    log_list.append(['precision=' + str(precision)])
    log_list.append(['recall=' + str(recall)])
    log_list.append(['f1=' + str(f1)])
    log_df = pd.DataFrame(log_list)
    log_df.to_csv(path_or_buf=os.path.join(result_dir, 'log'), header=False, index=False)
