# 第九章 搭建神经网络进行图像分类

## 9.1 实验数据准备

MIT67数据集，一个标准的室内场景检测数据集

http://web.mit.edu/torralba/www/indoor.html

## 9.2 数据预处理和准备

如何利用Pytorch构建需要的场景识别算法

### 9.2.1 数据集读取

### 9.2.2 重载data.Dataset类

### 9.2.3 transform数据预处理

## 9.3 模型构建

### 9.3.1 ResNet-50

### 9.3.2 bottleneck的实现

### 9.3.3 ResNet-50 卷积层定义

### 9.3.4 ResNet-50 forward实现

### 9.3.5 预训练参数装载

## 9.4 模型训练与结果评估

### 9.4.1 训练类的实现

### 9.4.2 优化器的定义

### 9.4.3 学习率衰减

### 9.4.4 训练

## 9.5 总结

ResNet-50详细介绍https://PyTorch.org/docs/stable/index.html



