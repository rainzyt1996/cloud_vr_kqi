import sys

from data_utils import DataUtils
import joblib
from lightgbm import LGBMClassifier
import logging
import numpy as np
import os
import pandas as pd
import program_logging
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import time

# from data_utils import DataUtils
from evaluation import evaluate, get_cross_var_score
import value


def test_adaboost(path_train_data: str,
                  path_test_data: str,
                  index_list: list,
                  params: dict,
                  dir_log_root: str,
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
    # params = {
    #     'n_estimators': np.arange(5, 105, 5),      # 50
    #     'learning_rate': np.arange(0.1, 1.1, 0.1),  # 1.0
    #     'max_depth': [None],          # None
    #     'min_samples_split': [2],   # 2
    #     'min_samples_leaf': [1],    # 1
    #     'max_features': [None]         # None
    # }

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
    file.write('[Model] AdaBoostClassifier\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column_train = ['Time_Train', 'Time_Predict',
                    'n_estimators', 'learning_rate',
                    'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'class_weight',
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
    for n_estimators in params['n_estimators']:
        for learning_rate in params['learning_rate']:
            learning_rate = np.around(learning_rate, 2)
            for max_depth in params['max_depth']:
                for min_samples_split in params['min_samples_split']:
                    for min_samples_leaf in params['min_samples_leaf']:
                        for max_features in params['max_features']:
                            for class_weight in params['class_weight']:
                                logging.info('Training...(n_estimators=' + str(n_estimators) +
                                             ',learning_rate=' + str(learning_rate) +
                                             ',max_depth=' + str(max_depth) +
                                             ',min_samples_split=' + str(min_samples_split) +
                                             ',min_samples_leaf=' + str(min_samples_leaf) +
                                             ',max_features=' + str(max_features) +
                                             ',class_weight=' + str(class_weight) + ')')
                                # 训练
                                time0 = time.time()
                                clf = AdaBoostClassifier(
                                    base_estimator=DecisionTreeClassifier(max_depth=max_depth,
                                                                          min_samples_split=min_samples_split,
                                                                          min_samples_leaf=min_samples_leaf,
                                                                          max_features=max_features,
                                                                          class_weight=class_weight),
                                    n_estimators=n_estimators,
                                    learning_rate=learning_rate)
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
                                res_train = [None, None,
                                             n_estimators, learning_rate,
                                             max_depth, min_samples_split, min_samples_leaf, max_features, class_weight]
                                res_train.extend(evaluate(train_labels, train_predict))
                                fTrain.write(','.join([str(x) for x in res_train]) + '\n')
                                time_train = time.time() - time0
                                # 预测
                                time0 = time.time()
                                test_predict = clf.predict(test_data)
                                # 计算预测结果
                                accuracy, precision, recall, f1, auc, fpr, tpr, thresholds = evaluate(test_labels, test_predict)
                                time_predict = time.time() - time0
                                # 记录预测结果
                                res = [time_train, time_predict,
                                       n_estimators, learning_rate,
                                       max_depth, min_samples_split, min_samples_leaf, max_features, class_weight,
                                       accuracy, precision, recall, f1, auc, fpr, tpr, thresholds] + cv_score
                                print('[Result]acc=' + str(accuracy) +
                                      ',pre=' + str(precision) +
                                      ',rec=' + str(recall) +
                                      ',f1=' + str(f1) +
                                      ',auc=' + str(auc) +
                                      ',fpr=' + str(fpr) +
                                      ',tpr=' + str(tpr))
                                file.write(','.join([str(x) for x in res]) + '\n')
                                # 保存结果
                                result = pd.DataFrame(data=test_predict, columns=['Predict'])
                                if type(class_weight) is dict:
                                    weight = class_weight[1]
                                else:
                                    weight = class_weight
                                filename = str(n_estimators) + '_' + \
                                           str(learning_rate) + '_' + \
                                           str(max_depth) + '_' + \
                                           str(min_samples_split) + '_' + \
                                           str(min_samples_leaf) + '_' + \
                                           str(max_features) + '_' + \
                                           str(weight)
                                result_filename = 'result_' + filename
                                result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
                                # 保存模型
                                model_filename = 'model_' + filename + '.pickle'
                                joblib.dump(value=clf, filename=os.path.join(dir_model, model_filename))
    file.close()
    fTrain.close()

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')


def test_gbdt(path_train_data: str,
              path_test_data: str,
              index_list: list,
              params: dict,
              dir_log_root: str,
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
    # params = {
    #     'n_estimators': np.arange(5, 105, 5),      # 50
    #     'learning_rate': np.arange(0.1, 1.1, 0.1),  # 1.0
    #     'max_depth': [None],          # None
    #     'min_samples_split': [2],   # 2
    #     'min_samples_leaf': [1],    # 1
    #     'max_features': [None]         # None
    # }

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
    file.write('[Model] GradientBoostingClassifier\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column_train = ['Time_Train', 'Time_Predict',
                    'n_estimators', 'learning_rate', 'subsample', 'max_depth', 'max_features',
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
    for n_estimators in params['n_estimators']:
        for learning_rate in params['learning_rate']:
            learning_rate = np.around(learning_rate, 2)
            for subsample in params['subsample']:
                subsample = np.around(subsample, 2)
                for max_depth in params['max_depth']:
                    for min_samples_split in params['min_samples_split']:
                        for min_samples_leaf in params['min_samples_leaf']:
                            for max_features in params['max_features']:
                                logging.info('Training...(n_estimators=' + str(n_estimators) +
                                             ',learning_rate=' + str(learning_rate) +
                                             ',subsample=' + str(subsample) +
                                             ',max_depth=' + str(max_depth) +
                                             ',max_features=' + str(max_features) + ')')
                                # 训练
                                time0 = time.time()
                                clf = GradientBoostingClassifier(n_estimators=n_estimators,
                                                                 learning_rate=learning_rate,
                                                                 subsample=subsample,
                                                                 max_depth=max_depth,
                                                                 min_samples_split=min_samples_split,
                                                                 min_samples_leaf=min_samples_leaf,
                                                                 max_features=max_features)
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
                                res_train = [None, None, n_estimators, learning_rate, subsample, max_depth, max_features]
                                res_train.extend(evaluate(train_labels, train_predict))
                                fTrain.write(','.join([str(x) for x in res_train]) + '\n')
                                time_train = time.time() - time0
                                # 预测
                                time0 = time.time()
                                test_predict = clf.predict(test_data)
                                # 计算预测结果
                                accuracy, precision, recall, f1, auc, fpr, tpr, thresholds = evaluate(test_labels, test_predict)
                                time_predict = time.time() - time0
                                # 记录预测结果
                                res = [time_train, time_predict,
                                       n_estimators, learning_rate, subsample, max_depth, max_features,
                                       accuracy, precision, recall, f1, auc, fpr, tpr, thresholds] + cv_score
                                print('[Result]acc=' + str(accuracy) +
                                      ',pre=' + str(precision) +
                                      ',rec=' + str(recall) +
                                      ',f1=' + str(f1) +
                                      ',auc=' + str(auc) +
                                      ',fpr=' + str(fpr) +
                                      ',tpr=' + str(tpr))
                                file.write(','.join([str(x) for x in res]) + '\n')
                                # 保存结果
                                result = pd.DataFrame(data=test_predict, columns=['Predict'])
                                filename = str(n_estimators) + '_' + \
                                           str(learning_rate) + '_' + \
                                           str(subsample) + '_' + \
                                           str(max_depth) + '_' + \
                                           str(min_samples_split) + '_' + \
                                           str(min_samples_leaf) + '_' + \
                                           str(max_features)
                                result_filename = 'result_' + filename
                                result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
                                # 保存模型
                                model_filename = 'model_' + filename + '.pickle'
                                joblib.dump(value=clf, filename=os.path.join(dir_model, model_filename))
    file.close()
    fTrain.close()

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')
    return


def test_gbdt_lgbm(path_train_data: str,
                   path_test_data: str,
                   index_list: list,
                   params: dict,
                   dir_log_root: str,
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
    file.write('[Model] LGBMClassifier(gbdt)\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column_train = ['Time_Train', 'Time_Predict',
                    'n_estimators', 'learning_rate', 'min_child_sample',
                    'subsample', 'max_depth', 'num_leaves', 'colsample_bytree',
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
    for n_estimators in params['n_estimators']:
        for learning_rate in params['learning_rate']:
            learning_rate = np.around(learning_rate, 2)
            for min_child_sample in params['min_child_sample']:
                for subsample in params['subsample']:
                    subsample = np.around(subsample, 2)
                    for max_depth in params['max_depth']:
                        for num_leaves in params['num_leaves']:
                            for colsample_bytree in params['colsample_bytree']:
                                logging.info('Training...(n_estimators=' + str(n_estimators) +
                                             ',learning_rate=' + str(learning_rate) +
                                             ',min_child_sample=' + str(min_child_sample) +
                                             ',subsample=' + str(subsample) +
                                             ',max_depth=' + str(max_depth) +
                                             ',num_leaves=' + str(num_leaves) +
                                             ',colsample_bytree=' + str(colsample_bytree) + ')')
                                # 训练
                                time0 = time.time()
                                clf = LGBMClassifier(n_estimators=n_estimators,
                                                     learning_rate=learning_rate,
                                                     min_child_samples=min_child_sample,
                                                     subsample=subsample,
                                                     max_depth=max_depth,
                                                     num_leaves=num_leaves,
                                                     colsample_bytree=colsample_bytree,
                                                     device_type='gpu')
                                clf.fit(train_data, train_labels)
                                # 交叉检验
                                if cv_on:
                                    cv_score = get_cross_var_score(estimator=clf, X=train_data, y=train_labels,
                                                                   scoring=['accuracy', 'precision', 'recall', 'f1',
                                                                            'roc_auc'],
                                                                   cv=10)
                                else:
                                    cv_score = []
                                # 记录训练结果
                                train_predict = clf.predict(train_data)
                                res_train = [None, None,
                                             n_estimators, learning_rate, min_child_sample,
                                             subsample, max_depth, num_leaves, colsample_bytree]
                                res_train.extend(evaluate(train_labels, train_predict))
                                fTrain.write(','.join([str(x) for x in res_train]) + '\n')
                                time_train = time.time() - time0
                                # 预测
                                time0 = time.time()
                                test_predict = clf.predict(test_data)
                                # 计算预测结果
                                accuracy, precision, recall, f1, auc, fpr, tpr, thresholds = evaluate(test_labels,
                                                                                                      test_predict)
                                time_predict = time.time() - time0
                                # 记录预测结果
                                res = [time_train, time_predict,
                                       n_estimators, learning_rate, min_child_sample,
                                       subsample, max_depth, num_leaves, colsample_bytree,
                                       accuracy, precision, recall, f1, auc, fpr, tpr, thresholds] + cv_score
                                print('[Result]acc=' + str(accuracy) +
                                      ',pre=' + str(precision) +
                                      ',rec=' + str(recall) +
                                      ',f1=' + str(f1) +
                                      ',auc=' + str(auc) +
                                      ',fpr=' + str(fpr) +
                                      ',tpr=' + str(tpr))
                                file.write(','.join([str(x) for x in res]) + '\n')
                                # 保存结果
                                result = pd.DataFrame(data=test_predict, columns=['Predict'])
                                filename = str(n_estimators) + '_' + \
                                           str(learning_rate) + '_' + \
                                           str(min_child_sample) + '_' + \
                                           str(subsample) + '_' + \
                                           str(max_depth) + '_' + \
                                           str(num_leaves) + '_' + \
                                           str(colsample_bytree)
                                result_filename = 'result_' + filename
                                result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
                                # 保存模型
                                model_filename = 'model_' + filename + '.pickle'
                                joblib.dump(value=clf, filename=os.path.join(dir_model, model_filename))
    file.close()
    fTrain.close()

    # 加载模型
    # clf = joblib.load('train/20210820011346/model.pickle')
    return


if __name__ == '__main__':
    # 分片路径
    # path_train_data = 'data/data_video/vall/ts_feature_label_v1_v10_0&1_1&2'
    # path_test_data = 'data/data_video/vall/ts_feature_label_v11_v13_0&1_1&2'
    # index_list = value.TS_FEATURES
    # dir_log_root = 'train/video/18_2stage/ts/RF'
    # 区间路径
    r = 5
    dr = 1
    path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
    path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
    index_list = value.REGION_FEATURE[r]
    # 参数
    cv_on = False
    if len(sys.argv) == 2:
        process = int(sys.argv[1])
    else:
        process = 0
    if process == 0:
        params = {
            'n_estimators': [60],  # 50
            'learning_rate': [0.6],  # 1.0
            'max_depth': [1, 2, 4],  # 1, 10-100
            'min_samples_split': [4, 6, 8],  # 2 +
            'min_samples_leaf': [8, 12, 16],  # 1 +
            'max_features': [None],  # None
            'class_weight': ['balanced']  # None, 'balanced', {0: , 1: }
        }
        dir_log_root = 'train/video/18_2stage/pkt/Adaboost/r' + str(r) + '/dr' + str(dr)
        test_adaboost(path_train_data=path_train_data,
                      path_test_data=path_test_data,
                      index_list=index_list,
                      params=params,
                      dir_log_root=dir_log_root,
                      cv_on=cv_on)
    elif process == 1:
        params = {
            'n_estimators': np.arange(1, 111, 10),  # 100
            'learning_rate': np.arange(0.2, 1.2, 0.2),  # 1
            'subsample': np.arange(0.5, 1.0, 0.2),   # 1
            'max_depth': [3],   # 3
            'min_samples_split': [2],  # 2
            'min_samples_leaf': [1],  # 1
            'max_features': [None]  # None, "auto"
        }
        dir_log_root = 'train/video/18_2stage/pkt/GBDT/r' + str(r) + '/dr' + str(dr)
        test_gbdt(path_train_data=path_train_data,
                  path_test_data=path_test_data,
                  index_list=index_list,
                  params=params,
                  dir_log_root=dir_log_root,
                  cv_on=cv_on)
    elif process == 2:
        params = {
            'n_estimators': [30, 50, 70],  # 100
            'learning_rate': [1.0],  # 0.1
            'min_child_sample': [10, 30, 50, 70],   # 20
            'subsample': [1],   # 1
            'max_depth': [6, 7],   # 3
            'num_leaves': [15, 31, 63, 127],     # 31 <2^max_depth
            'colsample_bytree': [1]     # 1
        }
        dir_log_root = 'train/video/18_2stage/pkt/GBDT_LGBM/r' + str(r) + '/dr' + str(dr)
        test_gbdt_lgbm(path_train_data=path_train_data,
                       path_test_data=path_test_data,
                       index_list=index_list,
                       params=params,
                       dir_log_root=dir_log_root,
                       cv_on=cv_on)
    else:
        logging.error('Invalid parameter!')
