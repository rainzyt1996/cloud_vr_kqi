import logging

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import program_logging
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
# from tsnecuda import TSNE


def example_tsne():
    """
    TSNE例程
    """
    '''降维过程'''
    x = np.array([[10, 56, 12], [80, 21, 92], [21, 30, 53], [11, 81, 15]])  # 数据(4X3)
    labels = np.array([1, 0, 1, 1])     # 每一行数据对应的标签(例如二分类问题)
    model = TSNE()
    np.set_printoptions(suppress=True)
    y = model.fit_transform(x)  # 将X降维(默认二维)后保存到Y中

    '''可视化过程'''
    plt.scatter(y[:, 0], y[:, 1], 20, labels)     # labels为每一行对应标签，20为标记大小
    # plt.savefig("transH.png")   # 保存图片
    plt.show()


def test_tsne():
    """
    TSNE测试
    """
    # 导入数据
    logging.info('Importing dataset...')
    path_data = 'data/data_20210723/data_video/data_video_1/data_video_1_1/ap_log_video_1_1/index_label'
    data_df = pd.read_csv(filepath_or_buffer=path_data)
    index_name_list = ['T1_T0_Sync', 'T2_T1_Sync', 'T0_100', 'Length', 'Retry']
    index = data_df[index_name_list].values
    label = data_df['Label'].values
    # 数据预处理
    logging.info('Preprocessing data...')
    scaler = MinMaxScaler()
    index = scaler.fit_transform(index)
    # t-SNE
    logging.info('t-SNE processing...')
    model = TSNE()
    np.set_printoptions(suppress=True)
    output = model.fit_transform(index)
    # 绘图并保存
    logging.info('Plotting...')
    plt.scatter(output[:, 0], output[:, 1], 10, label)
    path_save = 'data/data_20210723/data_video/data_video_1/data_video_1_1/analysis/tsne/tsne_mm.png'
    plt.savefig(path_save)
    plt.show()
