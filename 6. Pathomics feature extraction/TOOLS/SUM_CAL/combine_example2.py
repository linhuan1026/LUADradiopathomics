

import os
import pandas as pd


def find_csv():
    path_list = [x for x in os.listdir('/home/gzzstation/下载/real_pathomics/Code_Test/sum/')
                      if os.path.isfile(x) and os.path.splitext(x)[1] == '.csv']
    return path_list


if __name__ == '__main__':
    csvpath_list = find_csv()
    data = pd.DataFrame()

    for csv_file in csvpath_list:
        df = pd.read_csv(csv_file,encoding='gbk')
        df_data = pd.DataFrame(df)
        data = pd.concat([data,df_data])
    data.to_csv('./CAL_csv_汇总.csv',index = False,encoding='utf-8-sig')


