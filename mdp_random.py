# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

"""
函数说明：文件处理
Parameters:
     filename:数据文件
Returns:
     list_datasets：数据集特征列表
     category_labels:数据标签列表
"""
def data_handle(filename):
    read_data = pd.read_csv(filename)
    list_datasets = []
    category_labels = []
    for i in range(len(read_data)):
        list_data = []
        for j in range(len(read_data.iloc[i, :]) - 1):
            row_data = read_data.iloc[i, j]  # 读取每个样本的每个数据
            list_data.append(row_data)  #将每个数据存入列表
        list_datasets.append(list_data)  #将每个样本的数据存入列表

        row_data_label = read_data.iloc[i, len(read_data.iloc[i, :]) - 1]  # 读取每个样本的类别标签
        if row_data_label == 'N':
            category_labels.append(0)  # 将二分类标签转化为0和1,0代表软件正常，1代表软件缺陷
        else:
            category_labels.append(1)
    return list_datasets, category_labels

"""
函数说明：绘制ROC曲线
Parameters:
     labels:测试标签列表
     predict_prob:预测标签列表
"""
def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(labels, predict_prob)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)  #计算AUC值
    print('AUC=' + str(roc_auc))
    plt.title('PC5-ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.savefig('figures/PC5.png') #将ROC图片进行保存
    plt.show()


if __name__ == '__main__':
    datasets, labels = data_handle('MDP/KC4.csv')  # 对数据集进行处理

    # 训练集和测试集划分
    X_train = datasets[:115]
    y_train = labels[:115]
    X_test = datasets[90:]
    y_test = labels[90:]

    # 随机森林分类器
    clf = RandomForestClassifier()
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X_train, y_train)  # 使用训练集对分类器训练
    y_predict = clf.predict(X_test)  # 使用分类器对测试集进行预测

    print('准确率:', metrics.accuracy_score(y_test, y_predict)) #预测准确率输出
    print('宏平均精确率:',metrics.precision_score(y_test,y_predict,average='macro')) #预测宏平均精确率输出
    print('微平均精确率:', metrics.precision_score(y_test, y_predict, average='micro')) #预测微平均精确率输出
    print('宏平均召回率:',metrics.recall_score(y_test,y_predict,average='macro'))#预测宏平均召回率输出
    print('平均F1-score:',metrics.f1_score(y_test,y_predict,average='weighted'))#预测平均f1-score输出
    print('混淆矩阵输出:',metrics.confusion_matrix(y_test,y_predict))#混淆矩阵输出

    print('分类报告:', metrics.classification_report(y_test, y_predict))#分类报告输出
    plot_roc(y_test, y_predict)  #绘制ROC曲线并求出AUC值


