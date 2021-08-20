# encoding=utf-8
import logging
import time

# import numpy as np
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn import datasets
from sklearn import svm


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


def test_svm():
    logging.info('Preparing datasets...')
    path_data = 'data/data_20210723/data_video/data_video_1/data_video_1_1/ap_log_video_1_1/index_label'
    data_df = pd.read_csv(filepath_or_buffer=path_data)
    index_name_list = ['T1_T0_Sync', 'T2_T1_Sync', 'T0_100', 'Length', 'Retry']
    num_train = int(len(data_df) / 3 * 2)
    num_test = len(data_df) - num_train
    train_features = data_df[index_name_list].values[:num_train, :]
    train_labels = data_df['Label'].values[:num_train]
    test_features = data_df[index_name_list].values[num_train:num_train+num_test, :]
    test_labels = data_df['Label'].values[num_train:num_train+num_test]
    logging.info('Train data number: %d', num_train)
    logging.info('Test data number: %d', num_test)

    logging.info('Start training...')
    time0 = time.time()
    clf = svm.SVC()
    clf.fit(train_features, train_labels)
    time_train = time.time() - time0
    logging.info('Train complete. Time cost: %f s', time_train)

    # logging.info('Loading model...')
    # clf = joblib.load('train/20210820011346/model.pickle')

    logging.info('Start predicting...')
    time0 = time.time()
    test_predict = clf.predict(test_features)
    time_predict = time.time() - time0
    logging.info('Predict complete. Time cost: %f s', time_predict)

    score = accuracy_score(test_labels, test_predict)
    logging.info('The accuracy score is %f', score)

    logging.info('Saving model and log...')
    time_log = time.localtime()
    dir_log = 'train/' + time.strftime("%Y%m%d%H%M%S", time_log)
    os.mkdir(dir_log)
    joblib.dump(clf, dir_log + '/model.pickle')
    path_log = dir_log + '/train_log.txt'
    file = open(path_log, 'w')
    file.write('****** Train Log ******\n')
    file.write('Record time: ' + time.strftime("%Y-%m-%d %H:%M:%S", time_log) + '\n')
    file.write('*** Dataset ***\n')
    file.write('Dataset: data/data_20210723/data_video/data_video_1/data_video_1_1\n')
    file.write('Train data: data_video_1_1[0:' + str(num_train) + ']\n')
    file.write('Test data: data_video_1_1[' + str(num_train) + ':' + str(num_train+num_test) + ']\n')
    file.write('Index: ' + ', '.join(index_name_list) + '\n')
    file.write('*** Model ***\n')
    file.write('Model: svm.SVC()\n')
    file.write('*** Train ***\n')
    file.write('Time cost: ' + str(time_train) + ' s\n')
    file.write('*** Predict ***\n')
    file.write('Time cost: ' + str(time_predict) + ' s\n')
    file.write('Accuracy score: ' + str(score) + '\n')
    file.close()
