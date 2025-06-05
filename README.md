# 🧠 CNN 混合架构深度学习框架

[![Language](https://img.shields.io/badge/language-C%2B%2B-orange.svg)](https://isocpp.org/)
[![Language](https://img.shields.io/badge/language-python-blue.svg)](https://isocpp.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)](https://github.com/ApexGP/CNN_demo)

一个**高性能**的卷积神经网络框架，从零实现完整的深度学习流水线。结合 C/C++核心计算与 Python 易用接口，在 MNIST 数据集上达到**90.9%准确率**。

## 🏆 核心成果

### 🎯 卓越性能表现

- **🔥 最低 90.9% MNIST 准确率** - 通过多轮训练可以逐步提高准确率
- **🚀 高效实现** - C++核心 + Python 接口的最佳组合
- **📊 稳定训练** - 20 轮训练，损失从 2.28 降至 0.27

### 💻 技术架构亮点

```
✅ 完整反向传播算法
✅ 卷积层梯度计算
✅ MaxPool 层实现
✅ 交叉熵损失函数
✅ 真实 MNIST 数据集
✅ Xavier 参数初始化
✅ Dropout 正则化技术
✅ 多层网络架构
✅ SGD 优化器
✅ OpenMP 多线程加速
```

## 🚀 快速体验

### 30 秒快速开始

```bash
# 1. 克隆项目
git clone https://github.com/ApexGP/CNN_demo.git
cd CNN_demo

# 2. 设置环境变量 (可选，推荐)
scripts\setup_env.bat         # Windows
# source scripts/setup_env.sh  # Linux/macOS

# 3. 构建项目
build.bat --release --with-openblas --with-python

# 4. 运行演示程序
./build/bin/mnist_training.exe                    # C++演示
# python examples/python_examples/mnist_classifier.py  # Python演示 (Linux/macOS)
```

**智能路径解析** 🔍: 程序会自动查找数据文件，支持从项目根目录、build 目录或任意位置运行！

**期望输出：**

```
=== CNN混合架构演示: MNIST训练 ===
已加载 8000 个MNIST样本
网络参数数量: 63,658

开始训练...
轮次 20/20 - 训练损失: 0.268 - 训练准确率: 92.9%

测试结果 - 准确率: 90.9% ✨
```

## 🎯 核心特性

### 🔬 深度学习核心算法

- **卷积神经网络**：完整的 CNN 实现，包含卷积、池化、全连接层
- **反向传播算法**：从零实现的梯度计算和参数更新
- **正则化技术**：Dropout 防过拟合，提升泛化能力
- **优化算法**：SGD 优化器，支持动量和学习率调度

### ⚡ 高性能计算

- **C++核心引擎**：内存高效的张量操作和数学计算
- **OpenMP 并行**：多线程加速训练和推理
- **OpenBLAS 集成**：高性能线性代数运算
- **内存优化**：智能内存管理和缓存优化

### 🐍 Python 生态集成

- **pybind11 绑定**：无缝的 C++/Python 接口
- **NumPy 兼容**：直接支持 NumPy 数组操作
- **易用 API**：简洁直观的 Python API 设计

## 📊 性能基准

### MNIST 数字识别任务

| 指标           | 数值         | 说明             |
| -------------- | ------------ | ---------------- |
| **测试准确率** | **90.9%**    | 2000 个测试样本  |
| **训练准确率** | **94.4%**    | 8000 个训练样本  |
| **网络参数**   | **63,658**   | 高效的参数利用   |
| **训练时间**   | **16 分钟**  | 20 轮完整训练    |
| **收敛速度**   | **快速稳定** | 损失从 2.28→0.27 |

### 架构性能对比

| 网络配置             | 准确率    | 参数量     | 训练轮次  |
| -------------------- | --------- | ---------- | --------- |
| 基础 CNN             | 52.0%     | 2,572      | 5 轮      |
| 深度 CNN             | 89.9%     | 3,424      | 12 轮     |
| **最优配置+Dropout** | **90.9%** | **63,658** | **20 轮** |

## 🔧 技术架构

### 网络结构 (最优配置)

```cpp
// 90.9%准确率的获胜架构
network.add_conv_layer(8, 5, 1, 2);    // Conv: 1→8通道
network.add_relu_layer();
network.add_maxpool_layer(2, 2);       // Pool: 28×28→14×14
network.add_conv_layer(16, 5, 1, 0);   // Conv: 8→16通道
network.add_relu_layer();
network.add_maxpool_layer(2, 2);       // Pool: 14×14→7×7
network.add_flatten_layer();           // Flatten
network.add_fc_layer(128);             // FC: 128神经元
network.add_relu_layer();
network.add_dropout_layer(0.4f);       // Dropout: 40%
network.add_fc_layer(64);              // FC: 64神经元
network.add_relu_layer();
network.add_dropout_layer(0.3f);       // Dropout: 30%
network.add_fc_layer(10);              // Output: 10类别
```

### 关键优化技术

- **Xavier 初始化**：权重合理初始化，避免梯度消失
- **Dropout 正则化**：40%+30%丢弃率，防止过拟合
- **学习率调优**：0.02 最优学习率，平衡收敛速度与稳定性
- **数据增强**：8000 训练样本，充分的数据支持

## 💻 代码示例

### C++ API 使用

```cpp
#include "cnn/network.h"
#include "cnn/layers.h"

int main() {
    // 创建网络
    CNN::Network network;

    // 构建架构
    network.add_conv_layer(8, 5, 1, 2);
    network.add_relu_layer();
    network.add_maxpool_layer(2, 2);
    network.add_fc_layer(128);
    network.add_dropout_layer(0.4f);
    network.add_fc_layer(10);

    // 设置优化器
    network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.02f));
    network.set_loss_function(std::make_unique<CNN::CrossEntropyLoss>());

    // 训练
    network.train(train_images, train_labels, 20, 32, 0.02f);

    // 评估
    float accuracy = network.calculate_accuracy(test_images, test_labels);
    std::cout << "准确率: " << accuracy * 100 << "%" << std::endl;

    return 0;
}
```

### Python API 使用

```python
import cnn
import numpy as np

# 创建网络
net = cnn.Network()

# 添加层
net.add_conv_layer(8, 5, 1, 2)
net.add_relu_layer()
net.add_maxpool_layer(2, 2)
net.add_fc_layer(128)
net.add_dropout_layer(0.4)
net.add_fc_layer(10)

# 训练数据
X_train = np.random.randn(1000, 1, 28, 28).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# 训练
net.train(X_train, y_train, epochs=20, batch_size=32, lr=0.02)

# 预测
predictions = net.predict(X_test)
```

## 📦 安装和构建

### 系统要求

- **编译器**: GCC 7+, Clang 6+, MSVC 2019+
- **构建工具**: CMake 3.15+
- **依赖**: OpenBLAS, OpenMP, pybind11

### 快速安装

```bash
# Windows
build.bat --release --with-openblas --with-python

# Linux/macOS
./build.sh --release --with-openblas --with-python

# 检查依赖
python scripts/check_dependencies.py
```

详细安装指南请参考 → [**SETUP.md**](docs/SETUP.md)

## 📚 文档导航

| 文档                                        | 内容                    | 适合人群       |
| ------------------------------------------- | ----------------------- | -------------- |
| **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** | 框架设计原理和实现细节  | 开发者、研究者 |
| **[API_GUIDE.md](docs/API_GUIDE.md)**       | 完整 API 参考和使用指南 | 用户、集成者   |
| **[SETUP.md](docs/SETUP.md)**               | 安装配置和依赖管理      | 所有用户       |

## 🏗️ 项目结构

```
CNN_demo/
├── src/                        # 核心源代码
│   ├── core_c/                 # C核心计算库
│   ├── cpp/                    # C++面向对象封装
│   └── python/                 # Python绑定
├── include/                    # 头文件接口
├── examples/                   # 示例代码
│   └── cpp_examples/
│       └── mnist_training.cpp  # 90.9%准确率演示
├── tests/                      # 单元测试
├── docs/                       # 详细文档
├── build.bat                   # Windows构建脚本
├── build.sh                    # Linux构建脚本
└── CMakeLists.txt              # CMake构建配置
```

## 🧪 运行测试

```bash
# 构建并运行所有测试
build.bat --run-tests

# 运行特定测试
./build/bin/test_tensor
./build/bin/test_network
./build/bin/test_layers

# 运行MNIST演示
./build/bin/mnist_training
```

## 🚧 开发路线图

### 已完成 ✅

- [x] 完整的 CNN 架构实现
- [x] 90.9% MNIST 准确率
- [x] Dropout 正则化
- [x] 真实数据集训练
- [x] OpenMP 多线程加速
- [x] Python 绑定

### 开发中 🔄

- [ ] 批标准化(BatchNorm)优化
- [ ] Adam 优化器实现
- [ ] 数据增强技术
- [ ] CUDA GPU 加速

### 计划中 📋

- [ ] 更多数据集支持(CIFAR-10)
- [ ] 预训练模型
- [ ] 模型可视化工具
- [ ] 分布式训练

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- **OpenBLAS** - 高性能线性代数库
- **pybind11** - 优雅的 Python C++绑定
- **OpenMP** - 并行计算标准
- **MNIST 数据集** - 经典机器学习基准

---

<div align="center">

[⬆ 回到顶部](#-cnn-混合架构深度学习框架)

</div>
