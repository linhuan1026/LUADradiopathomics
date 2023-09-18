'''
统计tils
'''
import os
import csv
from scipy.io import loadmat, savemat
from glob import glob
from tqdm import tqdm
import pandas as pd
# Mat_path_x20_1='/media/peng/bb370be5-1ccb-40cb-a4ac-10fc0b308884/feng/TILS/20X/MAT'
# Mat_path_x40_1='//home/hero/DISK_14T/DATA/shanxi_156/process_44/post_process/MAT'
# Mat_path_x40_1='/media/gzzstation/14TB/PyRadiomics_Data/GD2/mat_40x/1138352-5.6/'


def gen_dict(x_lsit):
    path_x = {}
    WSI_list = []
    for one_path in x_lsit:
        WSI_ID = one_path.split('/')[-1]
        WSI_ID = WSI_ID.split('_')[0]
        if WSI_ID not in path_x:
            path_x[WSI_ID] = [one_path]
            WSI_list.append(WSI_ID)

        else:
            path_x[WSI_ID].append(one_path)
    return path_x, WSI_list


def gen_info(path_X40_dict, WSI_IDS, zoom='40x'):
    info = []
    for wsi_ID in tqdm(WSI_IDS):
        patch_list = path_X40_dict[wsi_ID]
        Patch_number = 0
        zoom = zoom
        All_inflam, All_neopla= 0, 0
        Mesenchyne_inflam, Tumor_inflam = 0, 0
        Connec_area, Tumor_area = 0, 0

        for patch_path in patch_list:
            if os.path.exists(patch_path):
                data = loadmat(patch_path)

                Patch_number += 1

                # All_connec += int(data['all_connec'][0][0])
                All_inflam += int(data['all_inflam'][0][0])
                All_neopla += int(data['all_neopla'][0][0])
                # All_other += int(data['all_other'][0][0])

                Mesenchyne_inflam += int(data['Mesenchyne_inflam'][0][0])
                Tumor_inflam += int(data['Tumor_inflam'][0][0])
                Connec_area += int(data['Connec_area'][0][0])
                Tumor_area += int(data['Tumor_area'][0][0])
                # lymph_area += int(data['lymph_area'][0][0])  # 这里会出错 
                # necrosis_area += int(data['necrosis_area'][0][0])
                # Inflam_inflam += int(data['Inflam_inflam'][0][0])

            else:
                print('failure')
                raise f'OPEN the {patch_path} failure'

        # print('patch_list:',len(patch_list))
        # print('Patch_number',Patch_number)
        info.append([wsi_ID, Patch_number, zoom, All_inflam, All_neopla,
                     Mesenchyne_inflam, Tumor_inflam, Connec_area, Tumor_area])

    return info


def run(Mat_path_x40_1, csv_name):
    # Header_line = ['ID', 'Patch_number', 'zoom', 'All_connec',
    #                'All_inflam', 'All_neopla', 'All_other',
    #                'Mesenchyne_inflam', 'Tumor_inflam', 'Connec_area',
    #                'Tumor_area', 'lymph_area', 'necrosis_area', 'Inflam_inflam']
    
    Header_line = ['ID', 'Patch_number', 'zoom','All_inflam', 
                'All_neopla','Mesenchyne_inflam', 'Tumor_inflam', 'Connec_area','Tumor_area']
    
    path_X40 = {}
    # path_X20={}

    X40_list_1 = glob(os.path.join(Mat_path_x40_1, '*.mat'))
    # X20_list_2 = glob(os.path.join(Mat_path_x20_1,'*.mat'))

    X40_list = X40_list_1
    # X20_list = X20_list_2
    path_X40_dict, WSI_IDS = gen_dict(X40_list)
    x40_info_lines = gen_info(path_X40_dict, WSI_IDS)

    # path_X20_dict,wsi_ids=gen_dict(X20_list)
    # x20_info_dict=gen_info(path_X20_dict,wsi_ids,zoom='20x')
    with open(csv_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(Header_line)
        # writer.writerows(x20_info_dict)
        writer.writerows(x40_info_lines)

def find_csv():
    path_list = [x for x in os.listdir('/home/gzzstation/下载/real_pathomics/Code_Test/sum/')
                      if os.path.isfile(x) and os.path.splitext(x)[1] == '.csv']
    return path_list


if __name__ == "__main__":

    #############################################################
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/GDPH/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/GD2/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/JMCH/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/SXCH/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/TCGA/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XYCH/mat_40x/'
    base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2/mat_40x/'
    # base_dir = '/media/gzzstation/14TB/PyRadiomics_Data/XY2-2/mat_40x/'

    center=base_dir.split('/')[-3]

    wsi_paths1 = os.listdir(base_dir)
    wsi_paths = sorted(wsi_paths1)
    length = len(wsi_paths)

    for wsi_path in wsi_paths:
        name = wsi_path.split('/')[-1]
        print('name:', name)

        csv_name=str(name+'.csv')

        Mat_path_x40_1 = base_dir+name+'/'
        run(Mat_path_x40_1, csv_name)
        print("### current sample calculate successfully ! ###")
        print("###")

        csv_name=str(name+'.csv')

        Mat_path_x40_1 = base_dir+name+'/'
        run(Mat_path_x40_1, csv_name)
    #############################################################

    # Mat_path_x40_1='/media/gzzstation/14TB/PyRadiomics_Data/GD2/算sun时出问题的mat/1309986/'
    # csv_name=str('1309986'+'.csv')
    # run(Mat_path_x40_1, csv_name)
    # print("###")


    csvpath_list = find_csv()
    data = pd.DataFrame()

    for csv_file in csvpath_list:
        df = pd.read_csv(csv_file,encoding='gbk')
        df_data = pd.DataFrame(df)
        data = pd.concat([data,df_data])
    data.to_csv(center+'_csv_汇总.csv',index = False,encoding='utf-8-sig')

