#!/usr/bin/env python
# -*- coding:utf-8 -*-
# datetime:2020/4/1 23:03

import os
import time
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from multiprocessing import cpu_count
from paddle.fluid.dygraph import Pool2D, Conv2D
from paddle.fluid.dygraph import Linear

# 生成图像列表
data_path = r'D:\CS\python\python_module\cv_baidu\day_002\Dataset'
character_folders = os.listdir(data_path)
# print(character_folders)
if (os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if (os.path.exists('./test_data.list')):
    os.remove('./test_data.list')

for character_folder in character_folders:

    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store':
                continue
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            count = 0
            for img in character_imgs:
                if img == '.DS_Store':
                    continue
                if count % 10 == 0:
                    f_test.write(os.path.join(data_path, character_folder, img) + '\t' + character_folder + '\n')
                else:
                    f_train.write(os.path.join(data_path, character_folder, img) + '\t' + character_folder + '\n')
                count += 1
print('列表已生成')


# 定义训练集和测试集的reader
def data_mapper(sample):
    img, label = sample
    img = Image.open(img)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    # 转置 rgb，rgb >> rr,gg,bb
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    return img, label


def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)

    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)


# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./train_data.list'), buf_size=256),
                            batch_size=32)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=32)

# 完成数据准备工作，开始训练网络

# 定义DNN网络
class MyDNN(fluid.dygraph.Layer):
    def __init__(self):
        super(MyDNN, self).__init__()
        self.hidden1 = Linear(100, 100, act='relu')
        self.hidden2 = Linear(100, 100, act='relu')
        self.hidden3 = Linear(100, 100, act='relu')
        self.hidden4 = Linear(3*100*100, 10, act='softmax')

    def forward(self, input):
        # print(input, shape)
        x = self.hidden1(input)
        # print(x.shape)
        x = self.hidden2(x)
        # print(x.shape)
        x = self.hidden3(x)
        x = fluid.layers.reshape(x, shape=[-1, 3*100*100])
        y = self.hidden4(x)
        # print(y.shape)

        return y


# 用动态图进行训练
with fluid.dygraph.guard():
    model = MyDNN()  # 模型实例化
    model.train()  # 训练模式
    opt = fluid.optimizer.SGDOptimizer(learning_rate=0.01,
                                       parameter_list=model.parameters())  # 优化器选用SGD随机梯度下降，学习率为0.001.

    epochs_num = 20  # 迭代次数

    for pass_num in range(epochs_num):

        for batch_id, data in enumerate(train_reader()):

            images = np.array([x[0].reshape(3, 100, 100) for x in data], np.float32)
            labels = np.array([x[1] for x in data]).astype('int64')
            labels = labels[:, np.newaxis]
            # print(images.shape)

            # numpy>pygraph
            image = fluid.dygraph.to_variable(images)
            label = fluid.dygraph.to_variable(labels)
            predict = model(image)  # 预测
            # print(predict)
            loss = fluid.layers.cross_entropy(predict, label)
            avg_loss = fluid.layers.mean(loss)  # 获取loss值

            acc = fluid.layers.accuracy(predict, label)  # 计算精度

            if batch_id != 0 and batch_id % 50 == 0:
                print(
                    "train_pass:{},batch_id:{},train_loss:{},train_acc:{}".format(pass_num, batch_id, avg_loss.numpy(),
                                                                                  acc.numpy()))

            avg_loss.backward() # 训练和测试只是差这一步
            opt.minimize(avg_loss)
            model.clear_gradients()

    fluid.save_dygraph(model.state_dict(), 'MyDNN')  # 保存模型

# 模型校验
with fluid.dygraph.guard():
    accs = []
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model = MyDNN()
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 训练模式
    for batch_id, data in enumerate(test_reader()):  # 测试集
        images = np.array([x[0].reshape(3, 100, 100) for x in data], np.float32)
        labels = np.array([x[1] for x in data]).astype('int64')
        labels = labels[:, np.newaxis]

        image = fluid.dygraph.to_variable(images)
        label = fluid.dygraph.to_variable(labels)

        predict = model(image)
        acc = fluid.layers.accuracy(predict, label)
        accs.append(acc.numpy()[0])
        avg_acc = np.mean(accs)
    print(avg_acc)


# 读取预测图像，进行预测

def load_image(path):
    img = Image.open(path)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    print(img.shape)
    return img


# 构建预测动态图过程
with fluid.dygraph.guard():
    infer_path = '111.JPG'
    model = MyDNN()  # 模型实例化
    model_dict, _ = fluid.load_dygraph('MyDNN')
    model.load_dict(model_dict)  # 加载模型参数
    model.eval()  # 评估模式
    infer_img = load_image(infer_path)
    infer_img = np.array(infer_img).astype('float32')
    infer_img = infer_img[np.newaxis, :, :, :]
    infer_img = fluid.dygraph.to_variable(infer_img)
    result = model(infer_img)
    display(Image.open('111.JPG'))
    print(np.argmax(result.numpy()))
