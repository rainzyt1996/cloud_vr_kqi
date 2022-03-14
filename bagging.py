from data_utils import DataUtils
import joblib
import logging
import numpy as np
import os
import pandas as pd
import program_logging
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from thundersvm import SVC
import time


def evaluate(labels, predict):
    accuracy = accuracy_score(labels, predict)
    precision = precision_score(labels, predict)
    recall = recall_score(labels, predict)
    f1 = f1_score(labels, predict)
    fpr, tpr, thresholds = roc_curve(labels, predict)
    # if len(fpr) >= 3:
    #     fpr = fpr[1]
    # else:
    #     fpr = 0
    # if len(tpr) >= 3:
    #     tpr = tpr[1]
    # else:
    #     tpr = 0
    # if len(thresholds) >= 3:
    #     thresholds = thresholds[1]
    # else:
    #     thresholds = 0
    return accuracy, precision, recall, f1, fpr, tpr, thresholds


def test_bagging(path_train_data: str, path_test_data: str, index_list: list):
    # 创建log路径
    time_log = time.localtime()
    dir_log = 'train/14_model/' + time.strftime("%Y%m%d%H%M%S", time_log)
    os.makedirs(name=dir_log, exist_ok=True)
    dir_result = dir_log + '/result_predict'
    os.makedirs(name=dir_result, exist_ok=True)

    # 导入数据集
    logging.info('Preparing datasets...')
    # 训练集
    file_df = pd.read_csv(filepath_or_buffer=path_train_data)
    data_df = file_df.loc[:, index_list]
    # data_avg = data_df.mean()
    # data_std = data_df.std()
    # data_df = (data_df - data_avg) / data_std
    train_data = data_df.values
    train_labels = file_df['Label'].values
    # 测试集
    file_df = pd.read_csv(filepath_or_buffer=path_test_data)
    data_df = file_df.loc[:, index_list]
    # data_df = (data_df - data_avg) / data_std
    test_data = data_df.values
    test_labels = file_df['Label'].values
    num_train = len(train_data)
    num_test = len(test_data)
    logging.info('Train data number: %d', num_train)
    logging.info('Test data number: %d', num_test)
    logging.info('Index list: ' + ', '.join(index_list))

    # 数据预处理
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # 配置训练参数
    params = {
        'n_estimators': [45],
        'max_depth': [45],
        'min_samples_split': [2],
        'min_samples_leaf': [9],
        'max_features': np.arange(1, 21, 2)
    }

    # 保存训练log
    path_log = dir_log + '/train_log.txt'
    file = open(path_log, 'w')
    file.write('****** Train Log ******\n')
    file.write('[Record time] ' + time.strftime("%Y-%m-%d %H:%M:%S", time_log) + '\n')
    file.write('*** Dataset ***\n')
    file.write('[Dataset] ' + os.path.dirname(os.path.dirname(path_train_data)) + '\n')
    file.write('[Train data] ' + str(num_train) + '\n')
    file.write('[Test data] ' + str(num_test) + '\n')
    file.write('[Index] ' + ', '.join(index_list) + '\n')
    file.write('[Preprocessing] ' + str(type(scaler)) + '\n')
    file.write('*** Model ***\n')
    file.write('[Model] BaggingClassifier(SVM)\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column = ['Time_Train', 'Time_Predict',
              'oob_score',
              'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'TPR', 'Thresholds']
    file = open(path_log, 'w')
    file.write(','.join([x for x in column]) + '\n')
    path_log2 = dir_log + '/result_train.csv'
    fTrain = open(path_log2, 'w')
    fTrain.write(','.join([x for x in column]) + '\n')

    logging.info('Training...(default)')
    # 训练
    time0 = time.time()
    svc = SVC(kernel='rbf', gpu_id=2)
    clf = BaggingClassifier(base_estimator=svc,
                            oob_score=True)
    clf.fit(train_data, train_labels)
    # 记录训练结果
    train_predict = clf.predict(train_data)
    res_train = []
    res_train.extend(evaluate(train_labels, train_predict))
    fTrain.write(','.join([str(x) for x in res_train]) + '\n')
    time_train = time.time() - time0
    # 预测
    time0 = time.time()
    test_predict = clf.predict(test_data)
    # 计算预测结果
    accuracy, precision, recall, f1, fpr, tpr, thresholds = evaluate(test_labels, test_predict)
    oob_score = clf.oob_score_
    time_predict = time.time() - time0
    # 记录预测结果
    res = [time_train, time_predict,
           oob_score,
           accuracy, precision, recall, f1, fpr, tpr, thresholds]
    print('[Result]oob_score=' + str(oob_score) +
          ',acc=' + str(accuracy) +
          ',pre=' + str(precision) +
          ',rec=' + str(recall) +
          ',f1=' + str(f1) +
          ',fpr=' + str(fpr) +
          ',tpr=' + str(tpr))
    file.write(','.join([str(x) for x in res]) + '\n')
    # 保存结果
    result = pd.DataFrame(data=test_predict, columns=['Predict'])
    result_filename = 'result_default'
    result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)

    file.close()
    fTrain.close()

    # # 保存模型
    # joblib.dump(clf, dir_log + '/model.pickle')

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')


if __name__ == '__main__':
    path_train_data = 'data/data_video/feature_label_v1_v9_1_1'
    path_test_data = 'data/data_video/feature_label_v10_v13_1_1'
    data_utils = DataUtils()
    index_list = data_utils.cFeature
    test_bagging(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list)
