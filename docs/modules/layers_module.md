# 神经网络层模块设计文档

## 概述

神经网络层模块实现了 CNN 中各种类型的层，包括卷积层、池化层、全连接层等。采用继承体系设计，提供统一的接口和高效的实现。

## 设计理念

### 1. 层次化继承体系

```
                    Layer (抽象基类)
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ConvLayer      ActivationLayer   PoolingLayer
        │               │               │
        │       ┌───────┼───────┐       ├─ MaxPoolLayer
        │       │       │       │       └─ AvgPoolLayer
        │   ReLULayer TanhLayer SigmoidLayer
        │
   FullyConnectedLayer            RegularizationLayer
                                        │
                                 ┌──────┼──────┐
                            DropoutLayer BatchNormLayer
```

### 2. 核心设计原则

- **统一接口**: 所有层都继承自`Layer`基类
- **前向后向分离**: 明确的前向传播和反向传播接口
- **状态管理**: 区分训练模式和推理模式
- **参数管理**: 自动管理权重和偏置参数
- **内存安全**: RAII 和智能指针管理资源

## 模块结构

### 基类 `Layer`

**文件位置**: `include/cnn/layers.h`

```cpp
class Layer {
public:
    virtual ~Layer() = default;

    // 核心接口
    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &grad_output) = 0;

    // 参数管理
    virtual std::vector<Tensor *> parameters() { return {}; }
    virtual std::vector<Tensor *> gradients() { return {}; }

    // 模式控制
    virtual void train(bool mode = true) { training_ = mode; }
    virtual bool is_training() const { return training_; }

    // 信息接口
    virtual std::string name() const = 0;
    virtual std::vector<int> output_shape(const std::vector<int> &input_shape) const = 0;

protected:
    bool training_ = true;
};
```

## 具体层实现

### 1. 卷积层 (ConvLayer)

**功能**: 执行 2D 卷积运算，CNN 的核心组件

**核心参数**:

- `in_channels`: 输入通道数
- `out_channels`: 输出通道数
- `kernel_size`: 卷积核大小
- `stride`: 步长
- `padding`: 填充

**实现要点**:

```cpp
class ConvLayer : public Layer {
private:
    int in_channels_, out_channels_;
    int kernel_size_, stride_, padding_;
    bool use_bias_;

    Tensor weights_;      // 形状: [out_channels, in_channels, kernel_size, kernel_size]
    Tensor bias_;         // 形状: [out_channels]
    Tensor weight_grad_;  // 权重梯度
    Tensor bias_grad_;    // 偏置梯度

    Tensor last_input_;   // 保存输入用于反向传播

public:
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;
};
```

**依赖关系**:

- **im2col 算法**: 将卷积转换为矩阵乘法
- **OpenBLAS**: 高效矩阵运算支持
- **OpenMP**: 多通道并行计算

### 2. 全连接层 (FullyConnectedLayer)

**功能**: 线性变换 y = Wx + b

**实现要点**:

```cpp
class FullyConnectedLayer : public Layer {
private:
    int in_features_, out_features_;
    bool use_bias_;

    Tensor weights_;     // 形状: [in_features, out_features]
    Tensor bias_;        // 形状: [out_features]
    Tensor last_input_;  // 保存输入

public:
    Tensor forward(const Tensor &input) override {
        last_input_ = input;
        // 实现: output = input.matmul(weights_) + bias_
        return input.matmul(weights_) + bias_;
    }

    Tensor backward(const Tensor &grad_output) override {
        // 计算权重梯度: weight_grad = input^T × grad_output
        weight_grad_ = last_input_.transpose().matmul(grad_output);

        // 计算偏置梯度: bias_grad = sum(grad_output, axis=0)
        bias_grad_ = grad_output.sum(0);

        // 计算输入梯度: input_grad = grad_output × weights^T
        return grad_output.matmul(weights_.transpose());
    }
};
```

### 3. 激活函数层

#### ReLU 层

```cpp
class ReLULayer : public Layer {
private:
    Tensor last_input_;

public:
    Tensor forward(const Tensor &input) override {
        last_input_ = input;
        return input.relu();  // max(0, x)
    }

    Tensor backward(const Tensor &grad_output) override {
        // ReLU梯度: grad_input[i] = grad_output[i] if input[i] > 0 else 0
        Tensor grad_input = grad_output.clone();
        const float* input_data = last_input_.data();
        float* grad_data = grad_input.data();

        for (size_t i = 0; i < grad_input.size(); ++i) {
            if (input_data[i] <= 0.0f) {
                grad_data[i] = 0.0f;
            }
        }
        return grad_input;
    }
};
```

#### Softmax 层

```cpp
class SoftmaxLayer : public Layer {
private:
    Tensor last_output_;
    int dim_;

public:
    Tensor forward(const Tensor &input) override {
        last_output_ = input.softmax();
        return last_output_;
    }

    Tensor backward(const Tensor &grad_output) override {
        // Softmax梯度计算较复杂，需要Jacobian矩阵
        return compute_softmax_gradient(last_output_, grad_output);
    }
};
```

### 4. 池化层

#### 最大池化层

```cpp
class MaxPoolLayer : public Layer {
private:
    int kernel_size_, stride_, padding_;
    Tensor last_input_;
    Tensor max_indices_;  // 保存最大值位置

public:
    Tensor forward(const Tensor &input) override {
        last_input_ = input;
        return maxpool2d(input, kernel_size_, stride_, padding_, max_indices_);
    }

    Tensor backward(const Tensor &grad_output) override {
        // 反向传播只传递到最大值位置
        return maxpool2d_backward(grad_output, max_indices_, last_input_.shape());
    }
};
```

### 5. 正则化层

#### Dropout 层

```cpp
class DropoutLayer : public Layer {
private:
    float p_;  // dropout概率
    Tensor dropout_mask_;

public:
    Tensor forward(const Tensor &input) override {
        if (training_) {
            // 训练模式：随机置零部分神经元
            dropout_mask_ = generate_dropout_mask(input.shape(), p_);
            return input * dropout_mask_ * (1.0f / (1.0f - p_));
        } else {
            // 推理模式：直接返回
            return input;
        }
    }

    Tensor backward(const Tensor &grad_output) override {
        if (training_) {
            return grad_output * dropout_mask_ * (1.0f / (1.0f - p_));
        } else {
            return grad_output;
        }
    }
};
```

#### 批量归一化层

```cpp
class BatchNormLayer : public Layer {
private:
    int num_features_;
    float eps_, momentum_;

    Tensor gamma_, beta_;           // 可学习参数
    Tensor gamma_grad_, beta_grad_; // 参数梯度
    Tensor running_mean_, running_var_;  // 移动平均

    // 反向传播临时变量
    Tensor last_input_, normalized_, std_;

public:
    Tensor forward(const Tensor &input) override {
        if (training_) {
            return batch_norm_training(input);
        } else {
            return batch_norm_inference(input);
        }
    }

    Tensor backward(const Tensor &grad_output) override {
        return batch_norm_backward(grad_output);
    }
};
```

## 依赖管理

### 第三方依赖

1. **OpenBLAS** (强烈推荐)

   - **影响层**: 卷积层、全连接层
   - **性能提升**: 矩阵运算加速 10-50 倍
   - **缺失处理**: 使用标准矩阵乘法，性能降低

2. **OpenMP** (推荐)
   - **影响层**: 所有层的并行计算
   - **性能提升**: 多线程加速 2-4 倍
   - **缺失处理**: 单线程执行

### 内部依赖关系

```
Layer接口
    ↓
各具体层实现
    ↓
Tensor运算
    ↓
math_core (数学函数)
    ↓
OpenBLAS + OpenMP (可选)
```

## 性能优化策略

### 1. 卷积层优化

- **im2col + GEMM**: 将卷积转换为高效矩阵乘法
- **Winograd 算法**: 小卷积核(3×3)的快速卷积
- **分组卷积**: 减少计算量
- **内存池**: 减少动态内存分配

### 2. 内存管理优化

- **原地操作**: 尽可能复用内存
- **延迟分配**: 按需分配梯度内存
- **内存对齐**: 利用 SIMD 指令集

### 3. 并行计算优化

- **数据并行**: 批量维度并行
- **通道并行**: 多通道同时计算
- **空间并行**: 输出空间位置并行

## 使用示例

### 构建网络

```cpp
#include "cnn/network.h"
#include "cnn/layers.h"

CNN::Network network;

// LeNet-5架构
network.add_conv_layer(6, 5, 1, 2);    // 6@28×28
network.add_relu_layer();
network.add_maxpool_layer(2, 2);       // 6@14×14
network.add_conv_layer(16, 5, 1, 0);   // 16@10×10
network.add_relu_layer();
network.add_maxpool_layer(2, 2);       // 16@5×5
network.add_flatten_layer();           // 400
network.add_fc_layer(120);
network.add_relu_layer();
network.add_fc_layer(84);
network.add_relu_layer();
network.add_fc_layer(10);
network.add_softmax_layer();
```

### 自定义层

```cpp
class CustomActivationLayer : public CNN::Layer {
public:
    Tensor forward(const Tensor &input) override {
        // 自定义激活函数: swish(x) = x * sigmoid(x)
        last_input_ = input;
        return input * input.sigmoid();
    }

    Tensor backward(const Tensor &grad_output) override {
        // swish'(x) = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        Tensor sigmoid_x = last_input_.sigmoid();
        Tensor grad = sigmoid_x + last_input_ * sigmoid_x * (1.0f - sigmoid_x);
        return grad_output * grad;
    }

    std::string name() const override { return "CustomActivation"; }

private:
    Tensor last_input_;
};
```

## 测试与验证

### 梯度检查

```cpp
bool gradient_check(Layer* layer, const Tensor& input) {
    const float eps = 1e-4f;
    const float tolerance = 1e-3f;

    // 前向传播
    Tensor output = layer->forward(input);
    Tensor grad_output = Tensor::ones_like(output);

    // 解析梯度
    Tensor analytical_grad = layer->backward(grad_output);

    // 数值梯度
    Tensor numerical_grad = compute_numerical_gradient(layer, input, eps);

    // 比较差异
    float max_diff = (analytical_grad - numerical_grad).abs().max();
    return max_diff < tolerance;
}
```

### 性能基准

```cpp
void benchmark_layer(Layer* layer, const Tensor& input, int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        Tensor output = layer->forward(input);
        Tensor grad = layer->backward(Tensor::ones_like(output));
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << layer->name() << " 平均耗时: "
              << duration.count() / iterations << " μs" << std::endl;
}
```

## 编译配置

### CMake 配置

```cmake
# 层模块依赖于张量模块
target_link_libraries(cnn_layers PRIVATE cnn_tensor)

# 可选加速库
if(USE_OPENBLAS)
    target_link_libraries(cnn_layers PRIVATE ${BLAS_LIBRARIES})
endif()

if(USE_OPENMP)
    target_link_libraries(cnn_layers PRIVATE OpenMP::OpenMP_CXX)
endif()
```

### 编译时优化

```cpp
// 编译时常量优化
template<int KernelSize, int Stride>
class OptimizedConvLayer : public ConvLayer {
    // 模板特化实现，编译时优化循环
};

// 条件编译
#ifdef USE_OPENBLAS
    // 使用BLAS优化路径
#else
    // 标准实现路径
#endif
```

## 未来改进方向

1. **更多层类型**

   - 空洞卷积 (Dilated Convolution)
   - 深度可分离卷积 (Depthwise Separable Convolution)
   - 注意力层 (Attention Layer)
   - Transformer 层

2. **量化支持**

   - INT8 量化推理
   - 混合精度训练

3. **图优化**

   - 算子融合 (Operator Fusion)
   - 内存优化
   - 计算图优化

4. **硬件加速**
   - CUDA GPU 支持
   - 专用 AI 芯片支持
