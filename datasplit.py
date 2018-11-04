
import pandas as pd


read_data=pd.read_csv('Pre_MDP/MC1.csv')

for i in range(len(read_data)):
    row_data_label = read_data.iloc[i, len(read_data.iloc[i, :]) - 1]  # 读取单个漏洞的类别标签
    if row_data_label == 'N':
        read_data.iloc[i,len(read_data.iloc[i, :]) - 1] =0 # 将单个漏洞的类别标签加入列表
    else:
        read_data.iloc[i, len(read_data.iloc[i, :]) - 1] =1

read_data[:100].to_csv('datatest/MC1.csv',index=False)
read_data[101:].to_csv('datatest/test_MC1.csv',index=False)
