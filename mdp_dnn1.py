# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import random
import numpy as np

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


"""
函数说明：深度神经网络模型(21-1000-1000-500-23)
Parameters:
    data_vec:数据集列表
    label_vec:标签列表
Returns: 
"""
def dnn_model(data_vec,label_vec):
    # 载入数据集并划分数据集为训练集和测试集
    trainset=data_vec[:9000]
    trainlabel=label_vec[:9000]
    testset=data_vec[9000:]
    testlabel=label_vec[9000:]

    # 设置批次的大小
    batch_size = 100
    n_batch=90

    # 定义批量梯度下降
    def next_batch(trainset, trainlabel, size):
        # random.sample()可以从指定的序列中，随机的截取指定长度的片断，存储索引号
        batchset = random.sample(range(len(trainset)), size)
        trainset_batch = trainset[batchset[0]]
        trainlabel_batch = trainlabel[batchset[0]]
        for i in range(1, size):
            trainset_batch = np.row_stack((trainset_batch, trainset[batchset[i]]))
            trainlabel_batch = np.row_stack((trainlabel_batch, trainlabel[batchset[i]]))
        return trainset_batch, trainlabel_batch

    # 定义初始化权值函数
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # 定义初始化偏置函数
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # 定义三个placeholder
    x = tf.placeholder(tf.float32, [None,21])
    y = tf.placeholder(tf.float32, [None, 2])
    keep_prob = tf.placeholder(tf.float32)  # 存放百分率

    # 创建一个多层神经网络模型
    # 第一个隐藏层
    W1 = weight_variable([21,500])
    b1 = bias_variable([500])
    L1 = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    L1_drop = tf.nn.dropout(L1, keep_prob)  # keep_prob设置工作状态神经元的百分率
    '''
    # 第二个隐藏层
    W2 = weight_variable([1000,1000])
    b2 = bias_variable([1000])
    L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
    L2_drop = tf.nn.dropout(L2, keep_prob)
    '''
    # 第三个隐藏层
    W3 = weight_variable([500,500])
    b3 = bias_variable([500])
    L3 = tf.nn.sigmoid(tf.matmul(L1_drop, W3) + b3)
    L3_drop = tf.nn.dropout(L3, keep_prob)
    # 输出层
    W4 = weight_variable([500,2])
    b4 = bias_variable([2])
    prediction = tf.nn.softmax(tf.matmul(L3_drop, W4) + b4)

    # 定义交叉熵代价函数
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    loss=tf.reduce_mean(tf.square(y-prediction))
    # 定义反向传播算法（使用梯度下降算法）
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 结果存放在一个布尔型列表中(argmax函数返回一维张量中最大的值所在的位置)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))

    # 求准确率(tf.cast将布尔值转换为float型)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 创建会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  # 初始化变量
        # 训练次数
        for i in range(15):
            for batch in range(n_batch):
                batch_xs, batch_ys = next_batch(trainset,trainlabel,batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
            # 测试数据计算出的准确率
            test_acc = sess.run(accuracy, feed_dict={x: testset, y: testlabel, keep_prob: 1.0})
            print("Iter" + str(i) + ",Testing Accuracy=" + str(test_acc))


if __name__=='__main__':
    datasets,labels=data_handle('Pre_MDP/CM1.csv')
    print(datasets)
    print(labels)
   # print('数据预处理成功')
   # print('使用深度神经网络进行训练')
   # dnn_model(datasets,labels)