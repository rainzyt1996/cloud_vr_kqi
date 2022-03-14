import joblib
import logging
import numpy as np
import os
import pandas as pd
import program_logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

# from data_utils import DataUtils
from evaluation import evaluate, get_cross_var_score
import value


def test_rf(path_train_data: str,
            path_test_data: str,
            index_list: list,
            params: dict,
            dir_log_root: str,
            cv_on=False,
            model_save=True):
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
    #     'n_estimators': np.arange(1, 201, 1),   # 100
    #     'max_depth': [None],      # None
    #     'min_samples_split': [2],   # 2
    #     'min_samples_leaf': [1],    # 1
    #     'max_features': ["auto"]     # "auto"
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
    file.write('[Model] RandomForestClassifier\n')
    file.write('[Parameter] \n')
    for pkey in params:
        file.write(' - ' + str(pkey) + ': ' + str(params[pkey]) + '\n')
    file.close()

    # 开始训练，保存训练结果
    logging.info('Start training...')
    path_log = dir_log + '/result_log.csv'
    column_train = ['Time_Train', 'Time_Predict',
                    'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'class_weight',
                    'oob_score',
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
        for max_depth in params['max_depth']:
            for min_samples_split in params['min_samples_split']:
                for min_samples_leaf in params['min_samples_leaf']:
                    for max_features in params['max_features']:
                        for class_weight in params['class_weight']:
                            logging.info('Training...(n_estimators=' + str(n_estimators) +
                                         ',max_depth=' + str(max_depth) +
                                         ',min_samples_split=' + str(min_samples_split) +
                                         ',min_samples_leaf=' + str(min_samples_leaf) +
                                         ',max_features=' + str(max_features) +
                                         ',class_weight=' + str(class_weight) + ')')
                            # 训练
                            time0 = time.time()
                            clf = RandomForestClassifier(n_estimators=n_estimators,
                                                         max_depth=max_depth,
                                                         min_samples_split=min_samples_split,
                                                         min_samples_leaf=min_samples_leaf,
                                                         max_features=max_features,
                                                         oob_score=True,
                                                         class_weight=class_weight)
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
                                         n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                                         class_weight,
                                         None]
                            res_train.extend(evaluate(train_labels, train_predict))
                            fTrain.write(','.join([str(x) for x in res_train]) + '\n')
                            time_train = time.time() - time0
                            # 预测
                            time0 = time.time()
                            test_predict = clf.predict(test_data)
                            # 计算预测结果
                            accuracy, precision, recall, f1, auc, fpr, tpr, thresholds = evaluate(test_labels, test_predict)
                            oob_score = clf.oob_score_
                            time_predict = time.time() - time0
                            # 记录预测结果
                            res = [time_train, time_predict,
                                   n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
                                   class_weight,
                                   oob_score,
                                   accuracy, precision, recall, f1, auc, fpr, tpr, thresholds] + cv_score
                            print('[Result]oob_score=' + str(oob_score) +
                                  ',acc=' + str(accuracy) +
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
                                       str(max_depth) + '_' + \
                                       str(min_samples_split) + '_' + \
                                       str(min_samples_leaf) + '_' + \
                                       str(max_features) + '_' + \
                                       str(class_weight)
                            result_filename = 'result_' + filename
                            result.to_csv(path_or_buf=os.path.join(dir_result, result_filename), index=False)
                            # 保存模型
                            if model_save:
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
    # dir_log_root = 'train/video/18_2stage/ts/RF'
    # 区间路径
    r = 5
    dr = 1
    path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
    path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
    index_list = value.REGION_FEATURE[r]
    dir_log_root = 'train/video/18_2stage/pkt/RF/r' + str(r) + '/dr' + str(dr)
    # 参数
    cv_on = False
    model_save = False
    params = {
        'n_estimators': [1, 2, 3],  # 100
        'max_depth': [3, 4, 5],  # None, 10-100
        'min_samples_split': [4, 5, 6],  # 2 +
        'min_samples_leaf': [13, 14, 15],  # 1 +
        'max_features': [None],  # "auto", None
        'class_weight': ['balanced']  # None, 'balanced', 'balanced_subsample'
    }
    # 训练
    test_rf(path_train_data=path_train_data,
            path_test_data=path_test_data,
            index_list=index_list,
            params=params,
            dir_log_root=dir_log_root,
            cv_on=cv_on,
            model_save=model_save)
