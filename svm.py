import copy

import joblib
import logging
# import numpy as np
import os
import pandas as pd
import program_logging
# from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from thundersvm import SVC
import time


def example_svm():

    print('prepare datasets...')
    # Iris数据集
    # iris=datasets.load_iris()
    # features=iris.data
    # labels=iris.target

    # MINST数据集
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]

    # 选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.33,
                                                                                random_state=0)

    time_2 = time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accuracy score is %f" % score)


def test_svm(test_index: str):
    # 创建log路径
    time_log = time.localtime()
    dir_log = 'train/7_t2_t0_test/only_t2_t0/' + time.strftime("%Y%m%d%H%M%S", time_log)
    os.mkdir(dir_log)
    dir_result = dir_log + '/result_predict'
    os.mkdir(dir_result)

    # 导入数据集
    logging.info('Preparing datasets...')
    path_data = 'data/data_20210723/data_video/data_video_1/data_video_1_1/ap_log_video_1_1/index_label_simple_test'
    data_df = pd.read_csv(filepath_or_buffer=path_data)
    index_name_list = ['T2_T0_Sync', test_index, 'Length', 'Retry']     # 'T1_T0_Sync', 'T2_T1_Sync', 'T0_100', 'Length', 'Retry'
    num_train = int(len(data_df) / 3 * 2)
    num_test = len(data_df) - num_train
    train_features = data_df.loc[:, index_name_list].values[:num_train, :]
    train_labels = data_df.loc[:, 'Label'].values[:num_train]
    test_features = data_df.loc[:, index_name_list].values[num_train:num_train+num_test, :]
    test_labels = data_df.loc[:, 'Label'].values[num_train:num_train+num_test]
    logging.info('Train data number: %d', num_train)
    logging.info('Test data number: %d', num_test)
    logging.info('Index list: ' + ', '.join(index_name_list))

    # 数据预处理
    scaler = MinMaxScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.fit_transform(test_features)

    # 配置训练参数
    kernel = 'rbf'   # linear, polynomial, rbf, sigmoid
    param_c = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]      # 1.0   [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    param_gamma = [0.001, 0.01, 0.1, 0.5, 1.0]     # 'auto'    [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
    param_degree = [3]      # 3 [1, 3, 5, 7, 9]

    # 保存训练log
    path_log = dir_log + '/train_log.txt'
    file = open(path_log, 'w')
    file.write('****** Train Log ******\n')
    file.write('[Record time] ' + time.strftime("%Y-%m-%d %H:%M:%S", time_log) + '\n')
    file.write('*** Dataset ***\n')
    file.write('[Dataset] data/data_20210723/data_video/data_video_1/data_video_1_1\n')
    file.write('[Train data] data_video_1_1[0:' + str(num_train) + ']\n')
    file.write('[Test data] data_video_1_1[' + str(num_train) + ':' + str(num_train+num_test) + ']\n')
    file.write('[Index] ' + ', '.join(index_name_list) + '\n')
    file.write('[Preprocessing] MinMaxScaler\n')
    file.write('*** Model ***\n')
    file.write('[Model] thundersvm.SVC()\n')
    file.write('[Parameter] \n')
    file.write('- kernel: ' + kernel + '\n')
    file.write('- C: ' + ', '.join([str(x) for x in param_c]) + '\n')
    file.write('- gamma: ' + ', '.join([str(x) for x in param_gamma]) + '\n')
    file.write('- degree: ' + ', '.join([str(x) for x in param_degree]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column = ['kernel', 'c', 'gamma', 'degree', 'Time_Train', 'Time_Predict', 'Accuracy', 'Precision', 'Recall', 'F1']
    result_info = []
    for c in param_c:
        for gamma in param_gamma:
            for degree in param_degree:
                logging.info('Training...(kernel='+kernel+',c='+str(c)+',gamma='+str(gamma)+',degree='+str(degree)+')')
                # 训练
                time0 = time.time()
                clf = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree)
                clf.fit(train_features, train_labels)
                time_train = time.time() - time0
                # 预测
                time0 = time.time()
                test_predict = clf.predict(test_features)
                time_predict = time.time() - time0
                # 计算结果
                accuracy = accuracy_score(test_labels, test_predict)
                precision = precision_score(test_labels, test_predict)
                recall = recall_score(test_labels, test_predict)
                f1 = f1_score(test_labels, test_predict)
                # 记录结果
                result_info.append([kernel, c, gamma, degree, time_train, time_predict, accuracy, precision, recall, f1])
                # # 保存结果
                result = pd.DataFrame(data=test_predict, columns=['Predict'])
                result_filename = 'result_' + kernel + '_' + str(c) + '_' + str(gamma) + '_' + str(degree)
                result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
    result_info_df = pd.DataFrame(data=result_info, columns=column)
    result_info_df.to_csv(path_or_buf=path_log, index=False)

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')

    # # 保存模型
    # joblib.dump(clf, dir_log + '/model.pickle')
