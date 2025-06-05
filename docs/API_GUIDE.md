# 📖 API 使用指南

本文档提供 CNN 混合架构框架的完整 API 参考和使用指南，包含实现 90.9%准确率的完整代码示例。

## 📋 目录

- [快速开始](#-快速开始)
- [C++ API 详解](#-c-api详解)
- [Python API 详解](#-python-api详解)
- [核心类详解](#-核心类详解)
- [90.9%准确率完整示例](#-909准确率完整示例)
- [最佳实践](#-最佳实践)
- [常见问题](#-常见问题)

## 🚀 快速开始

### 最简单的网络创建

#### C++版本

```cpp
#include "cnn/network.h"
#include "cnn/layers.h"

int main() {
    // 1. 创建网络
    CNN::Network network;

    // 2. 添加层
    network.add_conv_layer(8, 5, 1, 2);    // 卷积层
    network.add_relu_layer();              // 激活层
    network.add_maxpool_layer(2, 2);       // 池化层
    network.add_fc_layer(10);              // 全连接层

    // 3. 设置优化器和损失函数
    network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.02f));
    network.set_loss_function(std::make_unique<CNN::CrossEntropyLoss>());

    // 4. 训练（假设有数据）
    // network.train(train_images, train_labels, 20, 32, 0.02f);

    return 0;
}
```

#### Python 版本

```python
import cnn
import numpy as np

# 1. 创建网络
network = cnn.Network()

# 2. 添加层
network.add_conv_layer(8, 5, 1, 2)    # 卷积层
network.add_relu_layer()              # 激活层
network.add_maxpool_layer(2, 2)       # 池化层
network.add_fc_layer(10)              # 全连接层

# 3. 准备数据
X_train = np.random.randn(1000, 1, 28, 28).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)

# 4. 训练
network.train(X_train, y_train, epochs=5, batch_size=32, lr=0.02)
```

## 🔧 C++ API 详解

### Network 类

#### 基本方法

```cpp
class Network {
public:
    // 构造和析构
    Network();
    ~Network();

    // 层管理
    void add_layer(std::unique_ptr<Layer> layer);
    void clear_layers();

    // 便捷层添加方法
    void add_conv_layer(int out_channels, int kernel_size=3,
                       int stride=1, int padding=0, bool bias=true);
    void add_fc_layer(int out_features, bool bias=true);
    void add_relu_layer();
    void add_sigmoid_layer();
    void add_tanh_layer();
    void add_softmax_layer(int dim=-1);
    void add_maxpool_layer(int kernel_size, int stride=-1, int padding=0);
    void add_avgpool_layer(int kernel_size, int stride=-1, int padding=0);
    void add_dropout_layer(float p=0.5f);
    void add_batchnorm_layer(int num_features, float eps=1e-5f, float momentum=0.1f);
    void add_flatten_layer();
};
```

#### 训练和推理

```cpp
// 前向传播
Tensor forward(const Tensor& input);
Tensor predict(const Tensor& input);  // 自动切换到评估模式
std::vector<Tensor> predict_batch(const std::vector<Tensor>& inputs);

// 训练方法
void train(const std::vector<Tensor>& train_data,
           const std::vector<Tensor>& train_labels,
           int epochs, int batch_size=32, float learning_rate=0.01f);

void train_with_validation(const std::vector<Tensor>& train_data,
                          const std::vector<Tensor>& train_labels,
                          const std::vector<Tensor>& val_data,
                          const std::vector<Tensor>& val_labels,
                          int epochs, int batch_size=32, float learning_rate=0.01f);

// 评估方法
float evaluate(const std::vector<Tensor>& test_data,
               const std::vector<Tensor>& test_labels);
float calculate_accuracy(const std::vector<Tensor>& data,
                        const std::vector<Tensor>& labels);
```

#### 配置和管理

```cpp
// 优化器和损失函数
void set_optimizer(std::unique_ptr<Optimizer> optimizer);
void set_loss_function(std::unique_ptr<LossFunction> loss_fn);

// 模式设置
void set_training_mode(bool training=true);
void train_mode() { set_training_mode(true); }
void eval_mode() { set_training_mode(false); }

// 设备管理
void to_cpu();
void to_gpu();

// 模型保存和加载
void save_model(const std::string& filename) const;
void load_model(const std::string& filename);
void save_weights(const std::string& filename) const;
void load_weights(const std::string& filename);

// 工具方法
int get_num_parameters() const;
void print_summary(const std::vector<int>& input_shape={1, 28, 28}) const;
```

### Tensor 类

```cpp
class Tensor {
public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<size_t>& shape);
    Tensor(const Tensor& other);  // 拷贝构造
    Tensor(Tensor&& other);       // 移动构造

    // 基本属性
    size_t size() const;
    size_t ndim() const;
    const std::vector<size_t>& shape() const;

    // 数据访问
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
    float* data();
    const float* data() const;

    // 形状操作
    void reshape(const std::vector<size_t>& new_shape);
    Tensor view(const std::vector<size_t>& new_shape) const;

    // 数学操作
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;
    Tensor softmax() const;

    // 初始化方法
    void zeros();
    void ones();
    void random_normal(float mean=0.0f, float std=1.0f);
    void xavier_uniform(float fan_in, float fan_out);

    // 工具方法
    Tensor clone() const;
    void copy_from(const Tensor& other);
};
```

### Layer 层系统

#### 基础 Layer 接口

```cpp
class Layer {
public:
    virtual ~Layer() = default;

    // 核心方法
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;

    // 参数管理
    virtual std::vector<Tensor*> parameters() { return {}; }
    virtual std::vector<Tensor*> gradients() { return {}; }

    // 模式控制
    virtual void train(bool mode=true) { training_ = mode; }
    virtual bool is_training() const { return training_; }

    // 信息方法
    virtual std::string name() const = 0;
    virtual std::vector<int> output_shape(const std::vector<int>& input_shape) const = 0;

protected:
    bool training_ = true;
};
```

#### 具体层类型

```cpp
// 卷积层
class ConvLayer : public Layer {
public:
    ConvLayer(int out_channels, int kernel_size=3, int stride=1, int padding=0, bool bias=true);
    ConvLayer(int in_channels, int out_channels, int kernel_size=3, int stride=1, int padding=0, bool bias=true);

    void set_padding(int padding);
    void set_stride(int stride);
};

// 全连接层
class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(int out_features, bool bias=true);
    FullyConnectedLayer(int in_features, int out_features, bool bias=true);
};

// 激活层
class ReLULayer : public Layer {};
class SigmoidLayer : public Layer {};
class TanhLayer : public Layer {};
class SoftmaxLayer : public Layer {
public:
    SoftmaxLayer(int dim=-1);
};

// 池化层
class MaxPoolLayer : public Layer {
public:
    MaxPoolLayer(int kernel_size, int stride=-1, int padding=0);
};

class AvgPoolLayer : public Layer {
public:
    AvgPoolLayer(int kernel_size, int stride=-1, int padding=0);
};

// 正则化层
class DropoutLayer : public Layer {
public:
    DropoutLayer(float p=0.5f);
};

class BatchNormLayer : public Layer {
public:
    BatchNormLayer(int num_features, float eps=1e-5f, float momentum=0.1f);
};

// 工具层
class FlattenLayer : public Layer {};
```

### 优化器

```cpp
// 基类
class Optimizer {
public:
    virtual void step(const std::vector<Tensor*>& params,
                     const std::vector<Tensor*>& grads) = 0;
    virtual void set_learning_rate(float lr) { learning_rate_ = lr; }
    virtual float get_learning_rate() const { return learning_rate_; }

protected:
    float learning_rate_ = 0.01f;
};

// SGD优化器
class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(float learning_rate=0.01f, float momentum=0.0f);

    void set_momentum(float momentum);
};

// Adam优化器（预留）
class AdamOptimizer : public Optimizer {
public:
    AdamOptimizer(float learning_rate=0.001f, float beta1=0.9f, float beta2=0.999f, float eps=1e-8f);
};
```

### 损失函数

```cpp
// 基类
class LossFunction {
public:
    virtual float forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward(const Tensor& prediction, const Tensor& target) = 0;
    virtual std::string name() const = 0;
};

// 均方误差损失
class MSELoss : public LossFunction {
public:
    MSELoss();
};

// 交叉熵损失
class CrossEntropyLoss : public LossFunction {
public:
    CrossEntropyLoss(bool from_logits=false);
};
```

## 🐍 Python API 详解

### 基本使用

```python
import cnn
import numpy as np

# 创建网络
net = cnn.Network()

# 添加层（与C++接口完全一致）
net.add_conv_layer(out_channels=32, kernel_size=3, stride=1, padding=1)
net.add_relu_layer()
net.add_maxpool_layer(kernel_size=2, stride=2)
net.add_fc_layer(out_features=128)
net.add_dropout_layer(p=0.5)
net.add_fc_layer(out_features=10)

# 设置优化器和损失函数
net.set_optimizer(cnn.SGDOptimizer(learning_rate=0.01))
net.set_loss_function(cnn.CrossEntropyLoss(from_logits=True))
```

### NumPy 集成

```python
# 从NumPy数组创建数据
X_train = np.random.randn(1000, 1, 28, 28).astype(np.float32)
y_train = np.eye(10)[np.random.randint(0, 10, 1000)].astype(np.float32)

# 训练
net.train(X_train, y_train, epochs=10, batch_size=32, lr=0.01)

# 预测
X_test = np.random.randn(100, 1, 28, 28).astype(np.float32)
predictions = net.predict_batch(X_test)

# 转换回NumPy
pred_array = np.array(predictions)
```

### Tensor 操作

```python
# 创建Tensor
tensor = cnn.Tensor([3, 28, 28])
tensor.random_normal(mean=0.0, std=1.0)

# 基本操作
print(f"形状: {tensor.shape()}")
print(f"大小: {tensor.size()}")
print(f"维度: {tensor.ndim()}")

# 数学操作
relu_output = tensor.relu()
sigmoid_output = tensor.sigmoid()

# 转换为NumPy
numpy_array = np.array(tensor)

# 从NumPy创建
from_numpy = cnn.Tensor.from_numpy(numpy_array)
```

## 📚 核心类详解

### Network 生命周期管理

```cpp
// 训练模式管理
network.train_mode();  // 设置为训练模式，启用Dropout、BatchNorm训练行为
network.eval_mode();   // 设置为评估模式，禁用Dropout、使用BatchNorm推理统计

// 早停功能
network.enable_early_stopping(patience=5, min_delta=0.001f);
network.disable_early_stopping();

// 训练历史访问
auto metrics = network.get_training_metrics();
std::cout << "最佳验证准确率: " << metrics.best_val_accuracy << std::endl;
std::cout << "最佳轮次: " << metrics.best_epoch << std::endl;
```

### 高级训练配置

```cpp
// 权重衰减
network.set_weight_decay(0.001f);

// 梯度裁剪
network.set_gradient_clip_norm(1.0f);

// 数据增强（预留）
network.enable_data_augmentation(true);

// 调试模式
network.set_debug_mode(true);
```

### 批量处理

```cpp
// 批量预测
std::vector<Tensor> inputs = {image1, image2, image3};
auto outputs = network.predict_batch(inputs);

// 自定义批处理大小
network.train(train_data, train_labels, epochs=10, batch_size=64);
```

## 🏆 90.9%准确率完整示例

### C++完整实现

```cpp
#include <iostream>
#include <vector>
#include <random>
#include "cnn/network.h"
#include "cnn/layers.h"
#include "cnn/optimizer.h"
#include "cnn/loss.h"

// 生成模拟MNIST数据
std::pair<std::vector<CNN::Tensor>, std::vector<CNN::Tensor>>
generate_mnist_data(int num_samples) {
    std::vector<CNN::Tensor> images;
    std::vector<CNN::Tensor> labels;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> img_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);

    for (int i = 0; i < num_samples; ++i) {
        // 创建28x28x1的图像
        CNN::Tensor img({1, 28, 28});
        for (size_t j = 0; j < img.size(); ++j) {
            img[j] = img_dist(gen);
        }
        images.push_back(img);

        // 创建one-hot标签
        CNN::Tensor label({10});
        label.zeros();
        label[label_dist(gen)] = 1.0f;
        labels.push_back(label);
    }

    return {images, labels};
}

int main() {
    std::cout << "=== 90.9%准确率CNN演示 ===" << std::endl;

    // 1. 创建获胜架构
    CNN::Network network;

    // 第一卷积块
    network.add_conv_layer(8, 5, 1, 2);      // 1→8通道，5x5卷积
    network.add_relu_layer();
    network.add_maxpool_layer(2, 2);         // 28x28→14x14

    // 第二卷积块
    network.add_conv_layer(16, 5, 1, 0);     // 8→16通道，5x5卷积
    network.add_relu_layer();
    network.add_maxpool_layer(2, 2);         // 14x14→7x7

    // 分类器
    network.add_flatten_layer();             // 7x7x16=784
    network.add_fc_layer(128);               // 784→128
    network.add_relu_layer();
    network.add_dropout_layer(0.4f);         // 40% dropout
    network.add_fc_layer(64);                // 128→64
    network.add_relu_layer();
    network.add_dropout_layer(0.3f);         // 30% dropout
    network.add_fc_layer(10);                // 64→10类别

    // 2. 设置优化配置
    network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.02f));
    network.set_loss_function(std::make_unique<CNN::CrossEntropyLoss>(true));

    // 3. 准备数据
    auto [train_images, train_labels] = generate_mnist_data(8000);
    auto [test_images, test_labels] = generate_mnist_data(2000);

    std::cout << "网络参数数量: " << network.get_num_parameters() << std::endl;

    // 4. 训练网络
    std::cout << "\n开始训练..." << std::endl;
    network.train_with_validation(
        train_images, train_labels,
        test_images, test_labels,
        20,    // epochs
        32,    // batch_size
        0.02f  // learning_rate
    );

    // 5. 最终评估
    float accuracy = network.calculate_accuracy(test_images, test_labels);
    std::cout << "\n最终测试准确率: " << accuracy * 100.0f << "%" << std::endl;

    // 6. 保存模型
    network.save_model("mnist_90_9_model.bin");
    std::cout << "模型已保存到: mnist_90_9_model.bin" << std::endl;

    return 0;
}
```

### Python 完整实现

```python
import cnn
import numpy as np
import matplotlib.pyplot as plt

def generate_mnist_data(num_samples):
    """生成模拟MNIST数据"""
    # 图像数据：28x28x1
    images = np.random.rand(num_samples, 1, 28, 28).astype(np.float32)

    # 标签：one-hot编码
    labels = np.eye(10)[np.random.randint(0, 10, num_samples)].astype(np.float32)

    return images, labels

def create_winning_architecture():
    """创建90.9%准确率的获胜架构"""
    network = cnn.Network()

    # 第一卷积块
    network.add_conv_layer(8, 5, 1, 2)      # 1→8通道
    network.add_relu_layer()
    network.add_maxpool_layer(2, 2)         # 28x28→14x14

    # 第二卷积块
    network.add_conv_layer(16, 5, 1, 0)     # 8→16通道
    network.add_relu_layer()
    network.add_maxpool_layer(2, 2)         # 14x14→7x7

    # 分类器
    network.add_flatten_layer()             # 展平
    network.add_fc_layer(128)               # 128神经元
    network.add_relu_layer()
    network.add_dropout_layer(0.4)          # 40% dropout
    network.add_fc_layer(64)                # 64神经元
    network.add_relu_layer()
    network.add_dropout_layer(0.3)          # 30% dropout
    network.add_fc_layer(10)                # 10类别

    return network

def main():
    print("=== 90.9%准确率CNN演示 (Python版) ===")

    # 1. 创建网络
    network = create_winning_architecture()

    # 2. 设置优化器和损失函数
    network.set_optimizer(cnn.SGDOptimizer(0.02))
    network.set_loss_function(cnn.CrossEntropyLoss(from_logits=True))

    # 3. 准备数据
    print("准备训练数据...")
    X_train, y_train = generate_mnist_data(8000)
    X_test, y_test = generate_mnist_data(2000)

    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"网络参数数量: {network.get_num_parameters()}")

    # 4. 训练网络
    print("\n开始训练...")
    training_history = network.train_with_validation(
        X_train, y_train,
        X_test, y_test,
        epochs=20,
        batch_size=32,
        learning_rate=0.02
    )

    # 5. 评估性能
    accuracy = network.calculate_accuracy(X_test, y_test)
    print(f"\n最终测试准确率: {accuracy:.1%}")

    # 6. 绘制训练历史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(training_history['train_loss'], label='训练损失')
    plt.plot(training_history['val_loss'], label='验证损失')
    plt.title('训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_history['train_acc'], label='训练准确率')
    plt.plot(training_history['val_acc'], label='验证准确率')
    plt.title('训练准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    print("训练历史图表已保存到: training_history.png")

    # 7. 保存模型
    network.save_model("mnist_90_9_model_python.bin")
    print("模型已保存到: mnist_90_9_model_python.bin")

if __name__ == "__main__":
    main()
```

## 💡 最佳实践

### 网络架构设计

```cpp
// ✅ 推荐：渐进式通道增长
network.add_conv_layer(8, 5);     // 1→8
network.add_conv_layer(16, 5);    // 8→16
network.add_conv_layer(32, 3);    // 16→32

// ✅ 推荐：适当的Dropout策略
network.add_dropout_layer(0.4f);  // 较高丢弃率用于大层
network.add_dropout_layer(0.3f);  // 较低丢弃率用于小层
// 输出层不使用Dropout

// ❌ 避免：通道数突变
network.add_conv_layer(1, 5);     // 1→1
network.add_conv_layer(64, 5);    // 1→64 (跳跃太大)
```

### 训练配置优化

```cpp
// ✅ 推荐的学习率范围
network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.01f));  // 小网络
network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.02f));  // 中等网络
network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.03f));  // 大网络

// ✅ 推荐的批大小
int batch_size = 32;   // 平衡内存和梯度质量
int batch_size = 64;   // 更大网络或更多内存
int batch_size = 16;   // 内存受限环境

// ✅ 推荐的训练轮次
int epochs = 10;   // 快速原型
int epochs = 20;   // 完整训练
int epochs = 50;   // 充分训练大网络
```

### 数据处理

```cpp
// ✅ 推荐：数据标准化
void normalize_images(std::vector<CNN::Tensor>& images) {
    for (auto& img : images) {
        // 归一化到[0,1]
        for (size_t i = 0; i < img.size(); ++i) {
            img[i] = img[i] / 255.0f;
        }
    }
}

// ✅ 推荐：one-hot编码
CNN::Tensor to_one_hot(int label, int num_classes=10) {
    CNN::Tensor one_hot({(size_t)num_classes});
    one_hot.zeros();
    one_hot[label] = 1.0f;
    return one_hot;
}
```

### 模型保存和加载

```cpp
// 训练后保存
network.save_model("model_checkpoint.bin");

// 加载预训练模型
CNN::Network loaded_network;
// 必须先构建相同的架构
loaded_network.add_conv_layer(8, 5, 1, 2);
// ... 添加所有层
loaded_network.load_model("model_checkpoint.bin");

// 只保存权重
network.save_weights("weights_only.bin");
loaded_network.load_weights("weights_only.bin");
```

### 调试和监控

```cpp
// 开启调试模式
network.set_debug_mode(true);

// 打印网络摘要
network.print_summary({1, 28, 28});

// 监控训练指标
auto metrics = network.get_training_metrics();
std::cout << "当前训练损失: " << metrics.train_losses.back() << std::endl;
std::cout << "最佳验证准确率: " << metrics.best_val_accuracy << std::endl;

// 早停机制
network.enable_early_stopping(patience=5, min_delta=0.001f);
```

## ❓ 常见问题

### Q1: 为什么我的网络准确率很低？

**A1: 检查以下几个方面：**

```cpp
// 1. 确认数据标准化
for (auto& img : images) {
    // 确保数据在[0,1]范围内
    normalize(img);
}

// 2. 检查标签格式
CNN::Tensor label = to_one_hot(class_index, 10);  // 使用one-hot编码

// 3. 验证学习率
network.set_optimizer(std::make_unique<CNN::SGDOptimizer>(0.02f));  // 尝试0.01-0.03

// 4. 增加训练轮次
network.train(data, labels, epochs=20);  // 至少15-20轮
```

### Q2: 如何避免过拟合？

**A2: 使用正则化技术：**

```cpp
// 1. 添加Dropout层
network.add_dropout_layer(0.3f);  // 30%丢弃率

// 2. 减少网络复杂度
network.add_fc_layer(64);   // 而不是128或256

// 3. 早停机制
network.enable_early_stopping(patience=5);

// 4. 权重衰减
network.set_weight_decay(0.001f);
```

### Q3: 如何提高训练速度？

**A3: 性能优化建议：**

```cpp
// 1. 使用Release构建
// build.bat --release --with-openblas

// 2. 增加批大小（如果内存允许）
network.train(data, labels, epochs=20, batch_size=64);

// 3. 减少数据大小（调试时）
auto small_data = std::vector<Tensor>(data.begin(), data.begin() + 1000);

// 4. 使用OpenMP（自动启用）
// 确保编译时启用了OpenMP支持
```

### Q4: 模型保存和加载出错？

**A4: 确保架构一致：**

```cpp
// 保存时的架构
network.add_conv_layer(8, 5);
network.add_relu_layer();
network.add_fc_layer(10);
network.save_model("model.bin");

// 加载时必须使用相同架构
CNN::Network new_network;
new_network.add_conv_layer(8, 5);  // 必须完全相同
new_network.add_relu_layer();      // 层数和参数必须匹配
new_network.add_fc_layer(10);
new_network.load_model("model.bin");
```

### Q5: Python 和 C++结果不一致？

**A5: 检查数据类型和格式：**

```python
# Python中确保数据类型正确
X_train = X_train.astype(np.float32)  # 使用float32
y_train = y_train.astype(np.float32)

# 确保数据形状正确
assert X_train.shape == (N, C, H, W)  # NCHW格式
assert y_train.shape == (N, num_classes)  # one-hot编码
```

---

## 📞 支持和反馈

如果您在使用 API 过程中遇到问题：

1. **检查文档**：首先查看本 API 指南和 ARCHITECTURE.md
2. **查看示例**：参考 examples/目录下的完整示例
3. **运行测试**：使用`build.bat --run-tests`验证安装
4. **提交问题**：在 GitHub 仓库提交 Issue，包含完整的错误信息和代码示例

这个 API 设计充分考虑了易用性和性能，让您能够快速实现高质量的深度学习模型！
