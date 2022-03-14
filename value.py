# 常量

TS_FEATURES = [
    'T1_T0_Avg_TS', 'T1_T0_Std_TS',
    'T2_T1_Avg_TS', 'T2_T1_Std_TS',
    'T2_T0_Avg_TS', 'T2_T0_Std_TS',
    'Length_Avg_TS', 'Length_Std_TS',
    'Retry_Avg_TS', 'Retry_Std_TS',
    'Disorder_Avg_TS', 'Disorder_Std_TS',
    'DT0_Avg_TS', 'DT0_Std_TS',
    'DT1_Avg_TS', 'DT1_Std_TS',
    'DT2_Avg_TS', 'DT2_Std_TS',
    'PktRate_T0_TS', 'BitRate_T0_TS',
    'PktRate_T1_TS', 'BitRate_T1_TS',
    'PktRate_T2_TS', 'BitRate_T2_TS'
]
"""分片特征名称列表"""

REGION_FEATURE_NAME = [
    'T1_T0_Avg', 'T1_T0_Std',
    'T2_T1_Avg', 'T2_T1_Std',
    'T2_T0_Avg', 'T2_T0_Std',
    'Length_Avg', 'Length_Std',
    'Retry_Avg', 'Retry_Std',
    'Disorder_Avg', 'Disorder_Std',
    'DT0_Avg', 'DT0_Std',
    'DT1_Avg', 'DT1_Std',
    'DT2_Avg', 'DT2_Std'
]
"""区间特征名称前缀列表"""

REGION_FEATURE = {
    5: [name + '_5' for name in REGION_FEATURE_NAME],
    10: [name + '_10' for name in REGION_FEATURE_NAME],
    20: [name + '_20' for name in REGION_FEATURE_NAME],
    50: [name + '_50' for name in REGION_FEATURE_NAME],
    100: [name + '_100' for name in REGION_FEATURE_NAME],
    200: [name + '_200' for name in REGION_FEATURE_NAME],
    500: [name + '_500' for name in REGION_FEATURE_NAME],
    1000: [name + '_1000' for name in REGION_FEATURE_NAME],
    2000: [name + '_2000' for name in REGION_FEATURE_NAME],
}
"""区间特征名称列表"""
