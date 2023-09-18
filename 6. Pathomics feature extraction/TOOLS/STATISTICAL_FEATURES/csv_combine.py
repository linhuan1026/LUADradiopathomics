import pandas as pd

epi_GD=pd.read_csv('./tissue_XY2+XY2-2.csv')
stroma_GD=pd.read_csv('./tissue_汇总.csv')
# print(epi_GD)

csv_combine=pd.concat([epi_GD, stroma_GD])
csv_combine.to_csv('./tissue_汇总+XY2+XY2-2.csv', index=None)

print("successfully!")

