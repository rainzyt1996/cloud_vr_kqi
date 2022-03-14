import sys

import logging
import numpy as np

import value
from svm import test_svm
from rf import test_rf
from boosting import test_adaboost, test_gbdt, test_gbdt_lgbm


if __name__ == '__main__':
    if len(sys.argv) != 2:
        logging.error('Parameters number dismatch!')
    else:
        model = sys.argv[1]
        if model == 'svm':
            r = 500
            dr = 1
            path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
            path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
            index_list = value.REGION_FEATURE[r]
            dir_log_root = 'train/video/18_2stage/pkt/SVM/r' + str(r) + '/dr' + str(dr)
            cv_on = False
            params = {
                'kernel': ['rbf'],  # linear, polynomial, rbf, sigmoid
                'c': [0.1, 1.0, 10.0],  # 1.0   0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
                'gamma': [0.01, 0.1, 1.0],  # 'auto'    0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0
                'degree': [3],  # 3 [1, 3, 5, 7, 9]
                'class_weight': ['balanced']  # None, 'balanced'
            }
            gpu_id = 1
            test_svm(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list,
                     params=params, dir_log_root=dir_log_root, cv_on=cv_on, gpu_id=gpu_id)
        elif model == 'rf':
            r = 500
            dr = 1
            path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
            path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
            index_list = value.REGION_FEATURE[r]
            dir_log_root = 'train/video/18_2stage/pkt/RF/r' + str(r) + '/dr' + str(dr)
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
            test_rf(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list,
                    params=params, dir_log_root=dir_log_root, cv_on=cv_on, model_save=model_save)
        elif model == 'ada':
            r = 500
            dr = 1
            path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
            path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
            index_list = value.REGION_FEATURE[r]
            cv_on = False
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
            test_adaboost(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list,
                          params=params, dir_log_root=dir_log_root, cv_on=cv_on)
        elif model == 'gbdt':
            r = 500
            dr = 1
            path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
            path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
            index_list = value.REGION_FEATURE[r]
            cv_on = False
            params = {
                'n_estimators': np.arange(1, 111, 10),  # 100
                'learning_rate': np.arange(0.2, 1.2, 0.2),  # 1
                'subsample': np.arange(0.5, 1.0, 0.2),  # 1
                'max_depth': [3],  # 3
                'min_samples_split': [2],  # 2
                'min_samples_leaf': [1],  # 1
                'max_features': [None]  # None, "auto"
            }
            dir_log_root = 'train/video/18_2stage/pkt/GBDT/r' + str(r) + '/dr' + str(dr)
            test_gbdt(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list,
                      params=params, dir_log_root=dir_log_root, cv_on=cv_on)
        elif model == 'lgbm':
            r = 500
            dr = 1
            path_train_data = 'data/data_video/vall/region_feature_' + str(r) + '_v1_v10_0&1_1&2'
            path_test_data = 'data/data_video/vall/region_feature_' + str(r) + '_v11_v13_0&1_1&2'
            index_list = value.REGION_FEATURE[r]
            cv_on = False
            params = {
                'n_estimators': [95, 100, 105],  # 100
                'learning_rate': [1.0],  # 0.1
                'min_child_sample': [10, 20, 30],  # 20
                'subsample': [0.4, 0.5, 0.6],  # 1
                'max_depth': [4, 5],  # 3
                'num_leaves': [15, 31],  # 31 <2^max_depth
                'colsample_bytree': [1]  # 1
            }
            dir_log_root = 'train/video/18_2stage/pkt/GBDT_LGBM/r' + str(r) + '/dr' + str(dr)
            test_gbdt_lgbm(path_train_data=path_train_data, path_test_data=path_test_data, index_list=index_list,
                           params=params, dir_log_root=dir_log_root, cv_on=cv_on)
        else:
            logging.error('Invalid parameters!')
