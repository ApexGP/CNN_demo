#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MNIST分类器示例 - 使用CNN框架的Python接口

这个示例展示了如何使用CNN框架的Python接口构建和训练一个简单的
卷积神经网络来对MNIST手写数字进行分类。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# 添加模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))

# 导入CNN框架
try:
    import cnn_framework as cf
except ImportError:
    print("错误: 找不到cnn_framework模块。请确保已编译Python绑定。")
    print("编译命令: cmake .. -G \"MinGW Makefiles\" -DBUILD_PYTHON=ON")
    print("然后: cmake --build . -j4")
    sys.exit(1)

# 加载MNIST数据集
def load_mnist():
    """
    加载MNIST数据集。
    如果本地没有数据集，会自动下载。
    
    返回:
        (训练图像, 训练标签, 测试图像, 测试标签)
    """
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        
        print("加载MNIST数据集...")
        mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
        X = mnist.data.astype('float32') / 255.0  # 归一化到0-1
        y = mnist.target.astype('int')
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=10000, random_state=42)
        
        # 重塑为CNN输入格式: [N, C, H, W]
        X_train = X_train.reshape(-1, 1, 28, 28)
        X_test = X_test.reshape(-1, 1, 28, 28)
        
        print(f"数据集加载完成: {X_train.shape[0]}训练样本, {X_test.shape[0]}测试样本")
        return X_train, y_train, X_test, y_test
    
    except Exception as e:
        print(f"加载MNIST数据集时出错: {e}")
        print("尝试创建随机测试数据...")
        
        # 创建随机测试数据
        X_train = np.random.rand(1000, 1, 28, 28).astype('float32')
        y_train = np.random.randint(0, 10, size=1000)
        X_test = np.random.rand(200, 1, 28, 28).astype('float32')
        y_test = np.random.randint(0, 10, size=200)
        
        print("已创建随机测试数据")
        return X_train, y_train, X_test, y_test

def main():
    # 加载数据集
    X_train, y_train, X_test, y_test = load_mnist()
    
    # 创建网络
    print("创建CNN网络...")
    net = cf.Network()
    
    # 构建LeNet-5类似的架构
    net.add_conv_layer(6, 5, stride=1, padding=2)  # 输出: 6@28x28
    net.add_relu_layer()
    net.add_maxpool_layer(2)  # 输出: 6@14x14
    net.add_conv_layer(16, 5, stride=1, padding=0)  # 输出: 16@10x10
    net.add_relu_layer()
    net.add_maxpool_layer(2)  # 输出: 16@5x5
    net.add_flatten_layer()  # 输出: 400
    net.add_fc_layer(120)
    net.add_relu_layer()
    net.add_fc_layer(84)
    net.add_relu_layer()
    net.add_fc_layer(10)
    net.add_softmax_layer()
    
    # 配置训练
    net.set_optimizer("adam", learning_rate=0.001)
    net.set_loss("cross_entropy")
    
    # 训练网络
    print("开始训练...")
    start_time = time.time()
    
    # 使用小批量进行训练演示
    n_samples = min(10000, X_train.shape[0])  # 限制样本数量以加快演示
    history = net.train(
        X_train[:n_samples], 
        y_train[:n_samples], 
        epochs=5, 
        batch_size=32,
        validation_split=0.1,
        shuffle=True,
        verbose=True
    )
    
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f}秒")
    
    # 评估模型
    print("评估模型...")
    accuracy, loss = net.evaluate(X_test, y_test)
    print(f"测试集准确率: {accuracy:.4f}, 损失: {loss:.4f}")
    
    # 可视化训练过程
    print("生成训练可视化...")
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='训练准确率')
    plt.plot(history['val_accuracy'], label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mnist_training.png')
    print("已保存训练图表至 mnist_training.png")
    
    # 可视化一些预测结果
    print("可视化预测结果...")
    n_samples = 10
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_indices):
        img = X_test[idx].reshape(28, 28)
        pred = net.predict(X_test[idx:idx+1])[0]
        pred_label = np.argmax(pred)
        true_label = y_test[idx]
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"预测: {pred_label}\n实际: {true_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('mnist_predictions.png')
    print("已保存预测结果可视化至 mnist_predictions.png")
    
    # 保存模型
    model_path = 'mnist_model.bin'
    net.save(model_path)
    print(f"模型已保存至 {model_path}")
    
    print("示例完成!")

if __name__ == "__main__":
    main() 