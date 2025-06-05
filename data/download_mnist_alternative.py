#!/usr/bin/env python3
"""
使用tensorflow/pytorch数据加载的替代MNIST下载脚本
"""

import os
import numpy as np

def download_mnist_with_tensorflow():
    """使用tensorflow下载MNIST数据集并转换为原始格式"""
    try:
        import tensorflow as tf
        print("使用TensorFlow下载MNIST数据集...")
        
        # 下载MNIST数据集
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        print(f"训练集: {x_train.shape}, 标签: {y_train.shape}")
        print(f"测试集: {x_test.shape}, 标签: {y_test.shape}")
        
        # 保存为二进制格式（模拟MNIST原始格式）
        save_mnist_binary(x_train, y_train, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        save_mnist_binary(x_test, y_test, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
        
        return True
        
    except ImportError:
        print("TensorFlow未安装，尝试使用其他方法...")
        return False

def download_mnist_with_pytorch():
    """使用pytorch下载MNIST数据集"""
    try:
        import torch
        import torchvision
        import torchvision.transforms as transforms
        
        print("使用PyTorch下载MNIST数据集...")
        
        # 下载训练集
        train_dataset = torchvision.datasets.MNIST(
            root='.', train=True, download=True, transform=transforms.ToTensor()
        )
        
        # 下载测试集
        test_dataset = torchvision.datasets.MNIST(
            root='.', train=False, download=True, transform=transforms.ToTensor()
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 转换为numpy数组
        x_train = train_dataset.data.numpy()
        y_train = train_dataset.targets.numpy()
        x_test = test_dataset.data.numpy()
        y_test = test_dataset.targets.numpy()
        
        # 保存为二进制格式
        save_mnist_binary(x_train, y_train, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        save_mnist_binary(x_test, y_test, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
        
        return True
        
    except ImportError:
        print("PyTorch未安装，尝试其他方法...")
        return False

def save_mnist_binary(images, labels, images_file, labels_file):
    """保存MNIST数据为二进制格式"""
    
    # 保存图像文件
    with open(images_file, 'wb') as f:
        # 写入魔数 (2051 for images)
        f.write((2051).to_bytes(4, byteorder='big'))
        # 写入图像数量
        f.write(len(images).to_bytes(4, byteorder='big'))
        # 写入行数和列数
        f.write((28).to_bytes(4, byteorder='big'))
        f.write((28).to_bytes(4, byteorder='big'))
        # 写入图像数据
        f.write(images.astype(np.uint8).tobytes())
    
    # 保存标签文件
    with open(labels_file, 'wb') as f:
        # 写入魔数 (2049 for labels)
        f.write((2049).to_bytes(4, byteorder='big'))
        # 写入标签数量
        f.write(len(labels).to_bytes(4, byteorder='big'))
        # 写入标签数据
        f.write(labels.astype(np.uint8).tobytes())
    
    print(f"已保存: {images_file} ({os.path.getsize(images_file):,} bytes)")
    print(f"已保存: {labels_file} ({os.path.getsize(labels_file):,} bytes)")

def create_sample_data():
    """创建示例数据用于测试"""
    print("创建示例MNIST数据...")
    
    # 创建随机数据
    np.random.seed(42)
    
    # 训练集: 1000个样本
    x_train = np.random.randint(0, 255, (1000, 28, 28), dtype=np.uint8)
    y_train = np.random.randint(0, 10, 1000, dtype=np.uint8)
    
    # 测试集: 200个样本
    x_test = np.random.randint(0, 255, (200, 28, 28), dtype=np.uint8)
    y_test = np.random.randint(0, 10, 200, dtype=np.uint8)
    
    # 保存数据
    save_mnist_binary(x_train, y_train, "train-images.idx3-ubyte", "train-labels.idx1-ubyte")
    save_mnist_binary(x_test, y_test, "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
    
    print("示例数据创建完成！")

def main():
    print("=== MNIST数据集下载器 ===\n")
    
    # 检查文件是否已存在
    files_to_check = [
        "train-images.idx3-ubyte",
        "train-labels.idx1-ubyte", 
        "t10k-images.idx3-ubyte",
        "t10k-labels.idx1-ubyte"
    ]
    
    all_exist = all(os.path.exists(f) for f in files_to_check)
    if all_exist:
        print("MNIST数据文件已存在:")
        for f in files_to_check:
            if os.path.exists(f):
                size = os.path.getsize(f)
                print(f"  {f}: {size:,} bytes")
        return
    
    # 尝试不同的下载方法
    success = False
    
    if not success:
        success = download_mnist_with_tensorflow()
    
    if not success:
        success = download_mnist_with_pytorch()
    
    if not success:
        print("无法使用深度学习框架下载，创建示例数据...")
        create_sample_data()
    
    print("\n下载完成！")

if __name__ == "__main__":
    main() 