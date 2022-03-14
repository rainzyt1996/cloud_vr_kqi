import copy

import joblib
import logging
import numpy as np
import os
import pandas as pd
import program_logging
# from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from thundersvm import SVC
import time

# from data_utils import DataUtils
from evaluation import evaluate, get_cross_var_score
import value


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


def test_svm(path_train_data: str,
             path_test_data: str,
             index_list: list,
             params: dict,
             dir_log_root: str,
             gpu_id: int,
             cv_on=False):
    # 创建log路径
    time_log = time.localtime()
    dir_log = dir_log_root + '/' + time.strftime("%Y%m%d%H%M%S", time_log)
    os.makedirs(name=dir_log, exist_ok=True)
    dir_result = dir_log + '/result_predict'
    os.makedirs(name=dir_result, exist_ok=True)
    dir_model = dir_log + '/model'
    os.makedirs(name=dir_model, exist_ok=True)

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
    logging.info('[Train data] ' + str(num_train) + ' (' + path_train_data + ')')
    logging.info('[Test data] ' + str(num_test) + ' (' + path_test_data + ')')
    logging.info('[Index] ' + ', '.join(index_list))

    # 数据预处理
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    # 配置训练参数
    # kernel = 'rbf'   # linear, polynomial, rbf, sigmoid
    # param_c = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]      # 1.0   [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # param_gamma = [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]     # 'auto'    [0.001, 0.01, 0.1, 0.5, 1.0, 10.0]
    # param_degree = [3]      # 3 [1, 3, 5, 7, 9]

    # 保存训练log
    path_log = dir_log + '/train_log.txt'
    file = open(path_log, 'w')
    file.write('****** Train Log ******\n')
    file.write('[Record time] ' + time.strftime("%Y-%m-%d %H:%M:%S", time_log) + '\n')
    file.write('*** Dataset ***\n')
    file.write('[Train data] ' + str(num_train) + ' (' + path_train_data + ')\n')
    file.write('[Test data] ' + str(num_test) + ' (' + path_test_data + ')\n')
    file.write('[Index] ' + ', '.join(index_list) + '\n')
    file.write('[Preprocessing] ' + str(type(scaler)) + '\n')
    file.write('*** Model ***\n')
    file.write('[Model] SVM\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column_train = ['Time_Train', 'Time_Predict',
                    'kernel', 'c', 'gamma', 'degree', 'class_weight',
                    'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'FPR', 'TPR', 'Thresholds']
    if cv_on:
        column_cv = ['Time_Fit', 'Time_Score', 'Accuracy_CV', 'Precision_CV', 'Recall_CV', 'F1_CV', 'Auc_CV']
    else:
        column_cv = []
    column = column_train + column_cv
    file = open(path_log, 'w')
    file.write(','.join([x for x in column]) + '\n')
    path_log2 = dir_log + '/result_train.csv'
    fTrain = open(path_log2, 'w')
    fTrain.write(','.join([x for x in column_train]) + '\n')
    for kernel in params['kernel']:
        for c in params['c']:
            c = np.around(c, 6)
            for gamma in params['gamma']:
                if gamma != 'auto':
                    gamma = np.around(gamma, 6)
                for degree in params['degree']:
                    for class_weight in params['class_weight']:
                        logging.info('Training...(kernel=' + kernel +
                                     ',c=' + str(c) +
                                     ',gamma=' + str(gamma) +
                                     ',degree=' + str(degree) +
                                     ',class_weight=' + str(class_weight) + ')')
                        # 训练
                        time0 = time.time()
                        clf = SVC(kernel=kernel,
                                  C=c,
                                  gamma=gamma,
                                  degree=degree,
                                  class_weight=class_weight,
                                  gpu_id=gpu_id)
                        clf.fit(train_data, train_labels)
                        # 交叉检验
                        if cv_on:
                            cv_score = get_cross_var_score(estimator=clf, X=train_data, y=train_labels,
                                                           scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                                                           cv=10)
                        else:
                            cv_score = []
                        # 记录训练结果
                        train_predict = clf.predict(train_data)
                        res_train = [None, None, kernel, c, gamma, degree, class_weight]
                        res_train.extend(evaluate(train_labels, train_predict))
                        fTrain.write(','.join([str(x) for x in res_train]) + '\n')
                        time_train = time.time() - time0
                        # 测试
                        time0 = time.time()
                        test_predict = clf.predict(test_data)
                        # 计算测试结果
                        accuracy, precision, recall, f1, auc, fpr, tpr, thresholds = evaluate(test_labels, test_predict)
                        time_predict = time.time() - time0
                        # 记录结果
                        res = [time_train, time_predict,
                               kernel, c, gamma, degree, class_weight,
                               accuracy, precision, recall, f1, auc, fpr, tpr, thresholds] + cv_score
                        print('[Result]acc=' + str(accuracy)
                              + ',pre=' + str(precision)
                              + ',rec=' + str(recall)
                              + ',f1=' + str(f1)
                              + ',auc=' + str(auc)
                              + ',fpr=' + str(fpr)
                              + ',tpr=' + str(tpr))
                        file.write(','.join([str(x) for x in res]) + '\n')
                        # 保存结果
                        result = pd.DataFrame(data=test_predict, columns=['Predict'])
                        filename = str(kernel) + '_' + str(c) + '_' + str(gamma) + '_' + str(degree) + '_' + str(class_weight)
                        result_filename = 'result_' + filename
                        result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
                        # 保存模型
                        model_filename = 'model_' + filename + '.pickle'
                        joblib.dump(value=clf, filename=os.path.join(dir_model, model_filename))
    file.close()
    fTrain.close()

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')


if __name__ == '__main__':
    # 分片路径
    # path_train_data = 'data/data_video/vall/ts_feature_label_v1_v10_0&1_1&2'
    # path_test_data = 'data/data_video/vall/ts_feature_label_v11_v13_0&1_1&2'
    # index_list = value.TS_FEATURES
    # dir_log_root = 'train/video/18_2stage/ts/SVM'
    # 区间路径
    r = 500
    dr = 1
    path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
    path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
    index_list = value.REGION_FEATURE[r]
    dir_log_root = 'train/video/18_2stage/pkt/SVM/r' + str(r) + '/dr' + str(dr)
    # 参数
    cv_on = False
    params = {
        'kernel': ['rbf'],      # linear, polynomial, rbf, sigmoid
        'c': [0.1, 1.0, 10.0],             # 1.0   [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
        'gamma': [0.01, 0.1, 1.0],      # 'auto'    [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        'degree': [3],          # 3 [1, 3, 5, 7, 9]
        'class_weight': ['balanced']  # None, 'balanced'
    }
    gpu_id = 1
    test_svm(path_train_data=path_train_data,
             path_test_data=path_test_data,
             index_list=index_list,
             params=params,
             dir_log_root=dir_log_root,
             cv_on=cv_on,
             gpu_id=gpu_id)
