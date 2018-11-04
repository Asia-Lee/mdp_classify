# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import auc,roc_curve,accuracy_score,classification_report,precision_score,recall_score,f1_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split  #数据集划分
from sklearn.model_selection import KFold   #交叉验证
from sklearn.model_selection import StratifiedKFold #分层k折交叉验证
import math

"""
函数说明：文件处理
Parameters:
     filename:数据文件
Returns:
     list_datasets：分词后的数据集列表
     category_labels: 文本标签列表
"""
def data_handle(filename):
    read_data=pd.read_csv(filename)
    list_datasets=[]
    category_labels=[]
    for i in range(len(read_data)):
        list_data=[]
        for j in range(len(read_data.iloc[i,:])-1):
            row_data = read_data.iloc[i, j]           # 读取单个漏洞描述文本
            list_data.append(row_data)

        list_datasets.append(list_data)
        row_data_label=read_data.iloc[i,len(read_data.iloc[i,:])-1]   #读取单个漏洞的类别标签
        if row_data_label=='N':
            category_labels.append(0) #将单个漏洞的类别标签加入列表
        else:
            category_labels.append(1)
    return list_datasets, category_labels

def plot_roc(labels, predict_prob):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, predict_prob)
    roc_auc=auc(false_positive_rate, true_positive_rate)
    #print('AUC='+str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    #plt.savefig('figures/PC5.png')
    plt.show()



if __name__=='__main__':
    datasets,labels=data_handle('gf_dataset/CM1.csv')
    #x_train = datasets[:]
    #y_train = labels[:]
    #print(Counter(y_train))  # 查看所生成的样本类别分布
    #ros = RandomOverSampler(random_state=0)
    #x_resampled, y_resampled = ros.fit_sample(x_train, y_train)  #采用随机过采样
    #print(Counter(y_resampled))

    #x_test = datasets[400:]
    #y_test = labels[400:]
    #print(Counter(y_test))
    #rus=RandomUnderSampler(random_state=0)  #采用随机欠采样（下采样）
    #x_retest,y_retest=rus.fit_sample(x_test,y_test)
    #print(Counter(y_retest))

    '''
    x_train,x_test,y_train,y_test = train_test_split(datasets,labels,test_size=0.1,random_state=0) #数据集划分
    print(len(x_train))
    print(len(x_test))
    '''
    accuracy_list = []
    precision_list=[]
    recall_list=[]
    f1_list=[]
    auc_list=[]
    g_mean_list=[]
    balance_list=[]

   # kf=KFold(n_splits=10)
    kf=StratifiedKFold(n_splits=10,shuffle=True)
    for train_index,test_index in kf.split(datasets[:],labels[:]):
        rus=RandomUnderSampler(ratio=0.4,random_state=0,replacement=True) #采用随机欠采样（下采样）
        x_retest,y_retest=rus.fit_sample(datasets[:],labels[:])
        x_train=x_retest
        y_train=y_retest
        x_test=np.array(datasets)[test_index]
        y_test=np.array(labels)[test_index]



        clf = MLPClassifier(hidden_layer_sizes=(200), activation='tanh', solver='lbfgs', alpha=0.001,
                            batch_size=5, learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=50,
                            shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
                            nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                            beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)

        clf.fit(x_train, y_train)

        '''
        print(clf.n_iter_)
        print(clf.n_layers_)
        print(clf.n_outputs_)
        print(clf.batch_size)
        print(clf.loss)
        print(clf.activation)
        '''
        pre = clf.predict(x_test)
        #print('准确率：', accuracy_score(y_test, pre))
        #print('分类报告：', classification_report(y_test, pre))
        accuracy=accuracy_score(y_test,pre)
        precision=precision_score(y_test,pre,average='weighted')
        recall=recall_score(y_test,pre,average='weighted')
        f1score=f1_score(y_test,pre,average='weighted')
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pre)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        false_positive_rate, true_positive_rate=false_positive_rate[1], true_positive_rate[1]
        g_mean=math.sqrt(true_positive_rate*(1-false_positive_rate))
        balance=1-math.sqrt(math.pow((1-true_positive_rate),2)+math.pow((0-false_positive_rate),2))/math.sqrt(2)


        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1score)
        auc_list.append(roc_auc)
        g_mean_list.append(g_mean)
        balance_list.append(balance)

    print('k-accuracy:',accuracy_list)
    print('k-precision:',precision_list)
    print('k-recall:',recall_list)
    print('k-f1_score',f1_list)
    print('k-auc',auc_list)
    print('k-g_mean:',g_mean_list)
    print('k-balance:',balance_list)
    print('accuracy_ave:',np.mean(accuracy_list))
    print('precision_ave:',np.mean(precision_list))
    print('recall_ave:',np.mean(recall_list))
    print('f1_score_ave:',np.mean(f1_list))
    print('auc_ave:',np.mean(auc_list))
    print('g_mean_ave:',np.mean(g_mean_list))
    print('balance_ave:',np.mean(balance_list))





