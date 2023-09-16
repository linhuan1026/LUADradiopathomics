import os
import numpy as np
import glob
import cv2
import scipy.io as sio
from PIL import Image
import scipy
import scipy.ndimage
import xlwt
from xlwt import Workbook
from xlrd import open_workbook
import imageio
from xlutils.copy import copy
from multiprocessing import Process

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

# Compute Panoptic quality metric for each image
def Panoptic_quality(ground_truth_image, predicted_image):
    # print(predicted_image.shape)
    TP = 0
    FP = 0
    FN = 0
    sum_IOU = 0
    matched_instances = {}  # Create a dictionary to save ground truth indices in keys and predicted matched instances as velues
    # It will also save IOU of the matched instance in [indx][1]

    # Find matched instances and save it in a dictionary
    for i in np.unique(ground_truth_image):
        if i == 0:
            pass
        else:
            temp_image = np.array(ground_truth_image)
            temp_image = temp_image == i
            matched_image = temp_image * predicted_image

            for j in np.unique(matched_image):
                if j == 0:
                    pass
                else:
                    pred_temp = predicted_image == j
                    intersection = sum(sum(temp_image * pred_temp))
                    union = sum(sum(temp_image + pred_temp))
                    IOU = intersection / union
                    if IOU > 0.5:
                        # matched_instances[i] = j, IOU
                        matched_instances[i] = j, 1

                        # Compute TP, FP, FN and sum of IOU of the matched instances to compute Panoptic Quality

    pred_indx_list = np.unique(predicted_image)
    pred_indx_list = np.array(pred_indx_list[1:])

    # Loop on ground truth instances
    for indx in np.unique(ground_truth_image):
        if indx == 0:
            pass
        else:
            if indx in matched_instances.keys():
                pred_indx_list = np.delete(pred_indx_list, np.argwhere(pred_indx_list == matched_instances[indx][0]))
                TP = TP + 1
                sum_IOU = sum_IOU + matched_instances[indx][1]
            else:
                FN = FN + 1
    FP = len(np.unique(pred_indx_list))

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        F1 = 2*precision*recall/(precision+recall)
    # PQ = sum_IOU / (TP + 0.5 * FP + 0.5 * FN)
    # PQ_temp = PQ
    return F1


def main(i):
    import os
    num_of_group =i
    # rename model's name!
    this_model_name = "cca_from_0319_dice1_cs1_0415_2"
    iswatershed = True
    if iswatershed==True:
        iswatershed_name = "watershed"
    else:

        iswatershed_name = "no_watershed"

    ground_truth_path = '/home/cheng/workspace/hover_net-CCA_v0/metric_MoNuSAC/GT_32group_uint16_tif/group_' + str(i)  # Ground truth path to read data from

    create_folder("/home/cheng/workspace/hover_net-CCA_v0/metric_MoNuSAC_F1/wF1_results_xls_uint16_tif_revise0")
    path_save_xlsFiles = "/home/cheng/workspace/hover_net-CCA_v0/metric_MoNuSAC_F1/wF1_results_xls_uint16_tif_revise0/{:s}".format(this_model_name)
    create_folder(path_save_xlsFiles)
    path_save_xlsFiles_iswatershed = path_save_xlsFiles + "/{:s}_{:s}".format(this_model_name,iswatershed_name)
    create_folder(path_save_xlsFiles_iswatershed)
    os.chdir(ground_truth_path)  # the path to save xls files

    Predicted_path = '/home/cheng/workspace/hover_net-CCA_v0/metric_MoNuSAC/splited_for_metric_cca_from_0319_dice1_cs1_0415/cca_from_0319_dice1_cs1_0415_watershed/group_' + str(i)  # Path to read predicted outcomes from
    save_path = path_save_xlsFiles_iswatershed +'/{:s}_{:s}_{:02d}.xls'.format(this_model_name,iswatershed_name,i)

    # 自动生成 Predicted_path 多进程文件
    # p_path = '/home/cheng/workspace/hover_net-CCA_Los2/metric_MoNuSAC/Predicted_image_process'
    # if not os.path.exists(p_path):
    #     # 如果 Predicted_image_process 不存在，则创建一个
    #     os.makedirs("/home/cheng/workspace/hover_net-CCA_Los2/metric_MoNuSAC/Predicted_image_process")
    #


    # print("save_path")
    # print(save_path)
    # print(len(Predicted_path))
    # print((Predicted_path))



    participants_folders = glob.glob(Predicted_path + '/**')
    participants_folders.sort(reverse=True)
    # print(len(participants_folders)) # num of pictures
    # print(len(participants_folders))  # num of pictures
    # print((participants_folders))
    cell_types = ['Epithelial', 'Lymphocyte', 'Macrophage', 'Neutrophil']

    files = glob.glob('./**/**')
    files.sort()

    # files.reverse()
    # print("files")
    # print(files)
    len(files)  # Ground Truth files
    num_patients = len(files)
    # print("len(files)"+str(len(files)))
    # Workbook is created
    wb = Workbook()
    # wb.save(save_path)
    best_wF1 = 0
    best_wF1_dict = {}
    wF1_dict = {}
    # print(participants_folders)
    for participant_folder in participants_folders:

        # print(participant_folder[len(Predicted_path)+1:])  # To save only team name as an excel sheet instead of path of the folder containing predicted masks
        sheet_name = participant_folder[len(Predicted_path) + 1:]
        # print("sheet_name")
        # print(sheet_name)
        # add_sheet is used to create sheet of each participant
        ccbt = wb.add_sheet(sheet_name)
        ccbt.write(0, 0, 'Patient ID')
        ccbt.write(0, 1, 'Epithelial')
        ccbt.write(0, 2, 'Lymphocyte')
        ccbt.write(0, 3, 'Neutrophil')
        ccbt.write(0, 4, 'Macrophage')

        ccbt.write(0, 5, 'Epithelial')
        ccbt.write(0, 6, 'Lymphocyte')
        ccbt.write(0, 7, 'Neutrophil')
        ccbt.write(0, 8, 'Macrophage')

        ccbt.write(0, 9, 'wF1')

        # print(files)
        for image_count, filei in enumerate(files):
            ccbt.write(image_count + 1, 0, filei)  # Add image name in excel file
            # print(filei.split("/")[1])
            ## Ambiguous_region which was provided with the testing data to exclude from the metric computation

            imgs = glob.glob(filei + '/**/**')
            imgs.sort()

            ambiguous_idx = [i for i, f_name in enumerate(imgs) if 'Ambiguous' in f_name]

            # Check if abmiguous_idx exists
            if ambiguous_idx:
                ambiguous_regions = sio.loadmat(imgs[ambiguous_idx[0]])['n_ary_mask']
                ambiguous_regions = 1 - (ambiguous_regions > 0)
                imgs.pop(ambiguous_idx[0])

            for i, f_name in enumerate(imgs):
                # print(f_name)
                class_id = ([idx for idx in range(len(cell_types)) if cell_types[idx] in f_name])  # Cell-type

                # Read ground truth image
                # ground_truth = sio.loadmat(f_name)['n_ary_mask']
                ground_truth = imageio.imread(f_name)

                # Read predicted mask and exclude the ambiguous regions for metric computation
                pred_img_name = glob.glob(
                    participant_folder + '/' + filei.split("./")[1] + '/' + cell_types[class_id[0]] + '/**')
                pred_img_name.sort()
                # print(filei)
                if not pred_img_name:
                    ccbt.write(image_count + 1, class_id[0] + 1, 0)
                    ccbt.write(image_count + 1, class_id[0] + 1 + 4, 1)
                    # print(0)
                    # print("not pred_img_name")
                else:
                    # predicted_mask = sio.loadmat(pred_img_name[0])
                    predicted_mask = imageio.imread(pred_img_name[0])
                    # print('pred_img_name[0]')
                    # print(pred_img_name[0])
                    # mask_saved = ['img', 'name', 'n_ary_mask', 'Neutrophil_mask', 'arr', 'item', 'Epithelial_mask',
                    #               'Macrophage_mask', 'Lymphocyte_mask']
                    # mask_key = [m for m in mask_saved if m in predicted_mask.keys()]
                    # predicted_mask = predicted_mask[mask_key[0]]

                    # Converting binary to n-ary mask for those participants who did not send masks as informed
                    if (len(np.unique(predicted_mask)) == 2):
                        predicted_mask, num_features = scipy.ndimage.measurements.label(predicted_mask)

                    # print(pred_img_name)

                    if ambiguous_idx:
                        predicted_mask = predicted_mask * ambiguous_regions
                    predicted_mask2 = predicted_mask * 1

                    # Compute Panoptic Quality
                    # F1 = Panoptic_quality(ground_truth, predicted_mask)

                    F1 = Panoptic_quality(ground_truth, predicted_mask2)
                    # print(F1,pred_img_name)

                    ccbt.write(image_count + 1, class_id[0] + 1, F1)
                    ccbt.write(image_count + 1, class_id[0] + 1 + 4, 1)

        wb.save(save_path)

        read_data = open_workbook(save_path)
        table = read_data.sheet_by_name(sheet_name)
        sum_wF1 = 0
        count_wF1 = 0
        for i in range(1, num_patients + 1):
            # wcopy = copy(ccbt)
            Epithelial_val = table.cell(i, 1).value
            if isinstance(Epithelial_val, str):  # True
                if len(Epithelial_val) == 0:
                    Epithelial_val = '0'
            Epithelial_val = float(Epithelial_val)

            Lymphocyte_val = table.cell(i, 2).value

            if isinstance(Lymphocyte_val, str):  # True
                if len(Lymphocyte_val) == 0:
                    Lymphocyte_val = '0'
            Lymphocyte_val = float(Lymphocyte_val)

            Macrophage_val = table.cell(i, 3).value
            if isinstance(Macrophage_val, str):  # True
                if len(Macrophage_val) == 0:
                    Macrophage_val = '0'
            Macrophage_val = float(Macrophage_val)

            Neutrophil_val = table.cell(i, 4).value
            if isinstance(Neutrophil_val, str):  # True
                if len(Neutrophil_val) == 0:
                    Neutrophil_val = '0'
            Neutrophil_val = float(Neutrophil_val)

            Epithelial_weight = table.cell(i, 5).value
            if isinstance(Epithelial_weight, str):  # True
                if len(Epithelial_weight) == 0:
                    Epithelial_weight = '0'
            Epithelial_weight = float(Epithelial_weight)

            Lymphocyte_weight = table.cell(i, 6).value
            if isinstance(Lymphocyte_weight, str):  # True
                if len(Lymphocyte_weight) == 0:
                    Lymphocyte_weight = '0'
            Lymphocyte_weight = float(Lymphocyte_weight)

            Macrophage_weight = table.cell(i, 7).value
            if isinstance(Macrophage_weight, str):  # True
                if len(Macrophage_weight) == 0:
                    Macrophage_weight = '0'
            Macrophage_weight = float(Macrophage_weight)

            Neutrophil_weight = table.cell(i, 8).value
            if isinstance(Neutrophil_weight, str):  # True
                if len(Neutrophil_weight) == 0:
                    Neutrophil_weight = '0'
            Neutrophil_weight = float(Neutrophil_weight)

            wF1 = (Epithelial_val * 1 + Lymphocyte_val * 1 + Macrophage_val * 10 + Neutrophil_val * 10) / ((Epithelial_weight * 1 + Lymphocyte_weight * 1 + Macrophage_weight * 10 + Neutrophil_weight * 10) + 0.0000000001)
            # print('wF1')
            # print(wF1)
            # wcopy_file = copy(read_data)
            # wcopy_table = wcopy_file.get_sheet(participants_folders.index(participant_folder))
            sum_wF1 = sum_wF1 + wF1
            count_wF1 += 1
            ccbt.write(i, 9, wF1)

            """
            compute F1++i
            """
        for j in range(1, len(cell_types) + 1):
            now_class_val_sum = 0
            now_class_weight_sum = 0
            for i in range(1, num_patients + 1):
                now_cell_val = table.cell(i, j).value
                if isinstance(now_cell_val, str):  # True
                    if len(now_cell_val) == 0:
                        now_cell_val = '0'
                now_cell_val = float(now_cell_val)
                now_class_val_sum = now_class_val_sum + now_cell_val

                now_cell_weight = table.cell(i, j + 4).value
                if isinstance(now_cell_weight, str):  # True
                    if len(now_cell_weight) == 0:
                        now_cell_weight = '0'
                now_cell_weight = float(now_cell_weight)
                now_class_weight_sum = now_class_weight_sum + now_cell_weight
            F1Pluss_CLASSi = now_class_val_sum / now_class_weight_sum
            # print(F1Pluss_CLASSi)
            ccbt.write(num_patients + 2, j, F1Pluss_CLASSi)

        temp_wF1 = sum_wF1 / count_wF1
        wF1_dict[sheet_name] = temp_wF1
        if temp_wF1 > best_wF1:
            best_wF1 = temp_wF1
            best_wF1_dict = {sheet_name: best_wF1}
        ccbt.write(num_patients + 1, 9, sum_wF1 / count_wF1)
        # print("wF1_dict in group_{:d}".format(num_of_group))
        # print(wF1_dict)
        # print("best_wF1_dict in group_{:d}".format(num_of_group))
        # print(best_wF1_dict)
        wb.save(save_path)

    print("wF1_dict in group_{:d}".format(num_of_group))
    print(wF1_dict)
    print("best_wF1_dict in group_{:d}".format(num_of_group))
    print(best_wF1_dict)


# class MyProcess(Process):  # 继承Process类
#     def __init__(self):
#         super(MyProcess, self).__init__()
#
#     def run(self):
#         main()


if __name__ == "__main__":
    # 在这里使用多进程
    process_list = []
    for i in range(1, 33):  # 开启10个子进程执行函数
        p = Process(target=main, args=(i,))  # 实例化
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    # main()
    print("结束测试")






    # for i in range(i, num_patients + i):
    #     wF1_i  =
    """
    computer wF1
    """

"""
    新思路：
    把 Organizers_MoNuSAC_test_results_hover 文件夹里的150个单独的文件存放到10个文件中，对应10个进程
    新的思路：
    自动生成Predicted_image多进程文件夹
"""






