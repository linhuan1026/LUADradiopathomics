import random
import pandas as pd
import numpy as np
import os

from collections import defaultdict
sta_nuclei_feat = defaultdict(list)


# 当前sample的统计学特征
def sta_charcter_calculation(sample1_path, N, center, tissue):

    # 获取当前sample的名字,并加入到总表
    new_col_name = []

    sta_nuclei_feat['center'].append(center)
    new_col_name.append('center')

    sample_name1 = sample1_path.split('/')[-1]
    sample_name = sample_name1.split('_')[0]
    sta_nuclei_feat['patient_id'].append(sample_name)
    new_col_name.append('patient_id')

    sta_nuclei_feat['tissue'].append(tissue)
    new_col_name.append('tissue')

    sta_nuclei_feat['top'].append(N)
    new_col_name.append('top')

    print("获取当前sample的名字成功!并加入到总表")

    # 读取当前sample
    sample1 = pd.read_csv(sample1_path, header=0)

    # 获取所有列名
    col_names = sample1.columns.to_list()

    # 对于每一列的特征，分别计算五个统计学特征：
    for i in range(2, len(col_names)):
        current_column = sample1[col_names[i]].to_list()

        # # 随机选择10000个细胞
        # random.seed(0)
        # current_column_selected = random.sample(current_column, 10000)
        # current_column_selected = np.array(current_column_selected)

        # 选择当前样本的前50个patch，不够的就全取
        if len(sample1) <= N:
            current_column_selected = current_column
            current_column_selected = np.array(current_column_selected)

        else:
            current_column_selected = current_column[:N]
            current_column_selected = np.array(current_column_selected)

        # 计算mean
        cur_mean = np.mean(current_column_selected)

        # 计算中位数
        cur_median = np.median(current_column_selected)

        # 计算标准差
        cur_std = np.std(current_column_selected)

        # 计算四分位数间距
        quantile_low = np.percentile(current_column_selected, 25)
        quantile_high = np.percentile(current_column_selected, 75)
        cur_quantile = quantile_high-quantile_low

        # 计算变异系数
        cur_cov = cur_std/cur_mean

        cur_feat_mean = str(col_names[i])+'_mean'
        cur_feat_median = str(col_names[i])+'_median'
        cur_feat_std = str(col_names[i])+'_std'
        cur_feat_quantile = str(col_names[i])+'_quantile'
        cur_feat_cov = str(col_names[i])+'_cov'

        sta_nuclei_feat[cur_feat_mean].append(cur_mean)
        sta_nuclei_feat[cur_feat_median].append(cur_median)
        sta_nuclei_feat[cur_feat_std].append(cur_std)
        sta_nuclei_feat[cur_feat_quantile].append(cur_quantile)
        sta_nuclei_feat[cur_feat_cov].append(cur_cov)

        new_col_name.append(cur_feat_mean)
        new_col_name.append(cur_feat_median)
        new_col_name.append(cur_feat_std)
        new_col_name.append(cur_feat_quantile)
        new_col_name.append(cur_feat_cov)

    print('########### 当前样本的统计学特征计算成功! ###########')

    return sta_nuclei_feat, new_col_name


# 将当前的list保存成为csv文件
def save_list_to_csv(list_name, new_col_name, csv_save_path):
    sta_nuclei_feat = pd.DataFrame(list_name)
    sta_nuclei_feat.columns = new_col_name
    sta_nuclei_feat.to_csv(csv_save_path, encoding='gbk', index=None)
    print("################## save to csv successfully! ##################")


if __name__ == '__main__':

     ######## 需要设定的参数：########
    CSV_PATH = '/media/gzzstation/14TB/PyRadiomics_Results/example4/stroma_csv/SX_10x_Stroma_feat_csv/'
    center='SX'
    tissue='stroma'
    NUM = [50, 100, 150, 200, 250, 300, 350, 400]
    ######## 需要设定的参数：########
    
    for N in NUM:
        # SAVE_CSV_PATH = './sta_feat_'+tissue+'_'+center+'_top_'+str(N)+'.csv'
        SAVE_CSV_PATH = './sta_feat_'+tissue+'_'+center+'.csv'

        wsi_path = sorted(os.listdir(CSV_PATH))

        for csv_path in wsi_path:

            # 当前sample的路径：
            cur_sample_path = CSV_PATH+csv_path
            # 开始计算当前sample的统计学特征：
            final_list, final_col_name = sta_charcter_calculation(
                cur_sample_path, N, center, tissue)

        if N == NUM[-1]:
            save_list_to_csv(final_list, final_col_name, SAVE_CSV_PATH)
    
        print("############ process completed! ############")

