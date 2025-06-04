# 优化器模块设计文档

## 概述

优化器模块实现了深度学习中常用的优化算法，负责根据梯度信息更新网络参数。采用策略模式设计，支持多种优化算法。

## 设计理念

### 1. 策略模式设计

```
        Optimizer (抽象基类)
              │
    ┌─────────┼─────────┐
    │         │         │
SGDOptimizer  │  RMSpropOptimizer
    │         │         │
    │   AdamOptimizer   │
    │         │         │
    │   AdamWOptimizer  │
    │         │         │
    └─ AdagradOptimizer ┘
```

### 2. 核心设计原则

- **统一接口**: 所有优化器继承自`Optimizer`基类
- **无状态设计**: 优化器状态与参数分离
- **参数组支持**: 支持不同参数组使用不同超参数
- **梯度处理**: 集成梯度剪裁、权重衰减等功能
- **学习率调度**: 支持动态学习率调整

## 模块结构

### 基类 `Optimizer`

**文件位置**: `include/cnn/optimizer.h`

```cpp
namespace CNN {
class Optimizer {
public:
    virtual ~Optimizer() = default;

    // 核心接口
    virtual void step(const std::vector<Tensor*>& parameters,
                     const std::vector<Tensor*>& gradients) = 0;
    virtual void zero_grad(const std::vector<Tensor*>& gradients);

    // 配置接口
    virtual void set_learning_rate(float lr) { learning_rate_ = lr; }
    virtual float get_learning_rate() const { return learning_rate_; }

    // 状态管理
    virtual void reset_state() {}
    virtual std::string name() const = 0;

protected:
    float learning_rate_ = 0.001f;
    float weight_decay_ = 0.0f;
    float gradient_clip_norm_ = 0.0f;

    // 梯度处理辅助函数
    void apply_weight_decay(const std::vector<Tensor*>& parameters,
                           const std::vector<Tensor*>& gradients);
    void clip_gradients(const std::vector<Tensor*>& gradients);
};
}
```

## 具体优化器实现

### 1. SGD 优化器

**算法原理**: 随机梯度下降，最基础的优化算法

```
θ = θ - η * ∇θ
```

**带动量版本**:

```
v = β * v + ∇θ
θ = θ - η * v
```

**实现**:

```cpp
class SGDOptimizer : public Optimizer {
private:
    float momentum_;
    std::vector<Tensor> velocity_states_;  // 动量状态

public:
    SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.0f,
                 float weight_decay = 0.0f)
        : momentum_(momentum) {
        learning_rate_ = learning_rate;
        weight_decay_ = weight_decay;
    }

    void step(const std::vector<Tensor*>& parameters,
             const std::vector<Tensor*>& gradients) override {

        // 确保动量状态已初始化
        if (velocity_states_.size() != parameters.size()) {
            initialize_velocity_states(parameters);
        }

        for (size_t i = 0; i < parameters.size(); ++i) {
            if (!parameters[i] || !gradients[i]) continue;

            Tensor& param = *parameters[i];
            Tensor& grad = *gradients[i];
            Tensor& velocity = velocity_states_[i];

            if (momentum_ > 0.0f) {
                // 带动量的SGD
                velocity = velocity * momentum_ + grad;
                param = param - velocity * learning_rate_;
            } else {
                // 标准SGD
                param = param - grad * learning_rate_;
            }
        }
    }

    std::string name() const override { return "SGD"; }
};
```

### 2. Adam 优化器

**算法原理**: 自适应矩估计，结合动量和 RMSprop 的优点

**算法步骤**:

```
m_t = β₁ * m_{t-1} + (1-β₁) * ∇θ           // 一阶矩估计
v_t = β₂ * v_{t-1} + (1-β₂) * ∇θ²          // 二阶矩估计
m̂_t = m_t / (1-β₁^t)                        // 偏差修正
v̂_t = v_t / (1-β₂^t)                        // 偏差修正
θ = θ - η * m̂_t / (√v̂_t + ε)                // 参数更新
```

**实现**:

```cpp
class AdamOptimizer : public Optimizer {
private:
    float beta1_, beta2_, eps_;
    size_t step_count_;
    std::vector<Tensor> momentum_states_;     // 一阶矩
    std::vector<Tensor> variance_states_;     // 二阶矩

public:
    AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                  float beta2 = 0.999f, float eps = 1e-8f)
        : beta1_(beta1), beta2_(beta2), eps_(eps), step_count_(0) {
        learning_rate_ = learning_rate;
    }

    void step(const std::vector<Tensor*>& parameters,
             const std::vector<Tensor*>& gradients) override {

        step_count_++;
        initialize_states_if_needed(parameters);

        // 偏差修正系数
        float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
        float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

        for (size_t i = 0; i < parameters.size(); ++i) {
            if (!parameters[i] || !gradients[i]) continue;

            Tensor& param = *parameters[i];
            Tensor& grad = *gradients[i];
            Tensor& momentum = momentum_states_[i];
            Tensor& variance = variance_states_[i];

            // 更新一阶矩和二阶矩
            momentum = momentum * beta1_ + grad * (1.0f - beta1_);
            variance = variance * beta2_ + (grad * grad) * (1.0f - beta2_);

            // 偏差修正
            Tensor corrected_momentum = momentum * (1.0f / bias_correction1);
            Tensor corrected_variance = variance * (1.0f / bias_correction2);

            // 参数更新
            Tensor denominator = corrected_variance.sqrt() + eps_;
            param = param - corrected_momentum / denominator * learning_rate_;
        }
    }

    std::string name() const override { return "Adam"; }
};
```

### 3. AdamW 优化器

**算法原理**: Adam + 权重衰减解耦，改进的 Adam 算法

**关键改进**: 将权重衰减从梯度中分离，直接应用到参数上

```
θ = θ - η * (m̂_t / (√v̂_t + ε) + λ * θ)
```

**实现**:

```cpp
class AdamWOptimizer : public AdamOptimizer {
public:
    AdamWOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                   float beta2 = 0.999f, float eps = 1e-8f,
                   float weight_decay = 0.01f)
        : AdamOptimizer(learning_rate, beta1, beta2, eps) {
        weight_decay_ = weight_decay;
    }

    void step(const std::vector<Tensor*>& parameters,
             const std::vector<Tensor*>& gradients) override {

        // 先应用标准Adam更新
        AdamOptimizer::step(parameters, gradients);

        // 然后单独应用权重衰减
        if (weight_decay_ > 0.0f) {
            for (auto* param : parameters) {
                if (param) {
                    *param = *param * (1.0f - learning_rate_ * weight_decay_);
                }
            }
        }
    }

    std::string name() const override { return "AdamW"; }
};
```

### 4. RMSprop 优化器

**算法原理**: 自适应学习率，解决 Adagrad 学习率衰减过快的问题

**算法步骤**:

```
v_t = α * v_{t-1} + (1-α) * ∇θ²
θ = θ - η * ∇θ / (√v_t + ε)
```

**实现**:

```cpp
class RMSpropOptimizer : public Optimizer {
private:
    float alpha_, eps_, momentum_;
    std::vector<Tensor> variance_states_;
    std::vector<Tensor> momentum_states_;

public:
    RMSpropOptimizer(float learning_rate = 0.001f, float alpha = 0.99f,
                     float eps = 1e-8f, float momentum = 0.0f)
        : alpha_(alpha), eps_(eps), momentum_(momentum) {
        learning_rate_ = learning_rate;
    }

    void step(const std::vector<Tensor*>& parameters,
             const std::vector<Tensor*>& gradients) override {

        initialize_states_if_needed(parameters);

        for (size_t i = 0; i < parameters.size(); ++i) {
            if (!parameters[i] || !gradients[i]) continue;

            Tensor& param = *parameters[i];
            Tensor& grad = *gradients[i];
            Tensor& variance = variance_states_[i];

            // 更新梯度平方的移动平均
            variance = variance * alpha_ + (grad * grad) * (1.0f - alpha_);

            Tensor update = grad / (variance.sqrt() + eps_);

            if (momentum_ > 0.0f) {
                Tensor& momentum_buffer = momentum_states_[i];
                momentum_buffer = momentum_buffer * momentum_ + update;
                param = param - momentum_buffer * learning_rate_;
            } else {
                param = param - update * learning_rate_;
            }
        }
    }

    std::string name() const override { return "RMSprop"; }
};
```

### 5. Adagrad 优化器

**算法原理**: 累积梯度平方，为每个参数自适应学习率

**算法步骤**:

```
G_t = G_{t-1} + ∇θ²
θ = θ - η * ∇θ / (√G_t + ε)
```

**实现**:

```cpp
class AdagradOptimizer : public Optimizer {
private:
    float eps_;
    std::vector<Tensor> accumulated_gradients_;

public:
    AdagradOptimizer(float learning_rate = 0.01f, float eps = 1e-10f)
        : eps_(eps) {
        learning_rate_ = learning_rate;
    }

    void step(const std::vector<Tensor*>& parameters,
             const std::vector<Tensor*>& gradients) override {

        initialize_states_if_needed(parameters);

        for (size_t i = 0; i < parameters.size(); ++i) {
            if (!parameters[i] || !gradients[i]) continue;

            Tensor& param = *parameters[i];
            Tensor& grad = *gradients[i];
            Tensor& accumulated = accumulated_gradients_[i];

            // 累积梯度平方
            accumulated = accumulated + (grad * grad);

            // 参数更新
            param = param - grad / (accumulated.sqrt() + eps_) * learning_rate_;
        }
    }

    std::string name() const override { return "Adagrad"; }
};
```

## 高级功能

### 1. 学习率调度器

```cpp
class LearningRateScheduler {
public:
    virtual ~LearningRateScheduler() = default;
    virtual float get_lr(int epoch, float base_lr) = 0;
};

class StepLRScheduler : public LearningRateScheduler {
private:
    int step_size_;
    float gamma_;

public:
    StepLRScheduler(int step_size, float gamma = 0.1f)
        : step_size_(step_size), gamma_(gamma) {}

    float get_lr(int epoch, float base_lr) override {
        return base_lr * std::pow(gamma_, epoch / step_size_);
    }
};

class CosineAnnealingLRScheduler : public LearningRateScheduler {
private:
    int T_max_;
    float eta_min_;

public:
    CosineAnnealingLRScheduler(int T_max, float eta_min = 0.0f)
        : T_max_(T_max), eta_min_(eta_min) {}

    float get_lr(int epoch, float base_lr) override {
        return eta_min_ + (base_lr - eta_min_) *
               (1.0f + std::cos(M_PI * epoch / T_max_)) / 2.0f;
    }
};
```

### 2. 梯度剪裁

```cpp
class Optimizer {
protected:
    void clip_gradients(const std::vector<Tensor*>& gradients) {
        if (gradient_clip_norm_ <= 0.0f) return;

        // 计算梯度总范数
        float total_norm = 0.0f;
        for (const auto* grad : gradients) {
            if (grad) {
                const float* data = grad->data();
                for (size_t i = 0; i < grad->size(); ++i) {
                    total_norm += data[i] * data[i];
                }
            }
        }
        total_norm = std::sqrt(total_norm);

        // 如果超过阈值，按比例缩放
        if (total_norm > gradient_clip_norm_) {
            float scale = gradient_clip_norm_ / total_norm;
            for (auto* grad : gradients) {
                if (grad) {
                    *grad = *grad * scale;
                }
            }
        }
    }
};
```

### 3. 权重衰减

```cpp
class Optimizer {
protected:
    void apply_weight_decay(const std::vector<Tensor*>& parameters,
                           const std::vector<Tensor*>& gradients) {
        if (weight_decay_ <= 0.0f) return;

        for (size_t i = 0; i < parameters.size() && i < gradients.size(); ++i) {
            if (parameters[i] && gradients[i]) {
                // L2正则化：grad += weight_decay * param
                *gradients[i] = *gradients[i] + (*parameters[i] * weight_decay_);
            }
        }
    }
};
```

## 依赖关系

### 内部依赖

```
Optimizer接口
    ↓
各具体优化器实现
    ↓
Tensor运算 (加法、乘法、开方等)
    ↓
math_core (数学函数)
    ↓
OpenMP (可选，并行计算)
```

### 无第三方依赖

优化器模块主要依赖：

- **C++标准库**: `<cmath>`, `<vector>`, `<memory>`
- **Tensor 模块**: 用于参数和梯度操作
- **OpenMP**: 可选，用于并行计算加速

## 使用示例

### 基础使用

```cpp
#include "cnn/optimizer.h"
#include "cnn/network.h"

// 创建网络和优化器
CNN::Network network;
auto optimizer = std::make_unique<CNN::AdamOptimizer>(0.001f);

// 设置网络的优化器
network.set_optimizer(std::move(optimizer));

// 训练循环
for (int epoch = 0; epoch < 100; ++epoch) {
    for (const auto& batch : train_data) {
        // 前向传播
        Tensor output = network.forward(batch.input);

        // 计算损失
        float loss = loss_function.forward(output, batch.target);

        // 反向传播
        Tensor grad_output = loss_function.backward(output, batch.target);
        network.backward(grad_output);

        // 优化器更新参数
        auto params = network.parameters();
        auto grads = network.gradients();
        optimizer->step(params, grads);
        optimizer->zero_grad(grads);
    }
}
```

### 高级配置

```cpp
// 创建带权重衰减的AdamW优化器
auto optimizer = std::make_unique<CNN::AdamWOptimizer>(
    0.001f,  // learning_rate
    0.9f,    // beta1
    0.999f,  // beta2
    1e-8f,   // eps
    0.01f    // weight_decay
);

// 设置梯度剪裁
optimizer->set_gradient_clip_norm(1.0f);

// 创建学习率调度器
auto scheduler = std::make_unique<StepLRScheduler>(30, 0.1f);

// 训练过程中调整学习率
for (int epoch = 0; epoch < 100; ++epoch) {
    float new_lr = scheduler->get_lr(epoch, 0.001f);
    optimizer->set_learning_rate(new_lr);

    // ... 训练代码
}
```

### 不同参数组使用不同优化器

```cpp
// 为不同层设置不同的学习率
auto conv_optimizer = std::make_unique<CNN::SGDOptimizer>(0.01f, 0.9f);
auto fc_optimizer = std::make_unique<CNN::AdamOptimizer>(0.001f);

// 手动管理参数更新
auto conv_params = network.get_conv_parameters();
auto fc_params = network.get_fc_parameters();

conv_optimizer->step(conv_params.first, conv_params.second);
fc_optimizer->step(fc_params.first, fc_params.second);
```

## 性能优化

### 1. 内存优化

- **状态重用**: 避免重复分配优化器状态
- **原地操作**: 尽可能使用原地运算
- **内存池**: 预分配状态缓冲区

### 2. 计算优化

- **向量化**: 利用 SIMD 指令集
- **并行化**: OpenMP 并行参数更新
- **融合操作**: 将多个操作合并

```cpp
// 优化的Adam更新（融合操作）
void optimized_adam_update(Tensor& param, const Tensor& grad,
                          Tensor& momentum, Tensor& variance,
                          float lr, float beta1, float beta2, float eps) {
    #pragma omp parallel for
    for (size_t i = 0; i < param.size(); ++i) {
        float g = grad[i];

        // 更新动量和方差
        momentum[i] = beta1 * momentum[i] + (1.0f - beta1) * g;
        variance[i] = beta2 * variance[i] + (1.0f - beta2) * g * g;

        // 计算更新量
        float m_hat = momentum[i] / (1.0f - beta1);
        float v_hat = variance[i] / (1.0f - beta2);

        // 更新参数
        param[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
}
```

## 测试与验证

### 收敛性测试

```cpp
void test_optimizer_convergence() {
    // 简单的二次函数优化
    CNN::Tensor x({1});
    x[0] = 5.0f;  // 初始值

    auto optimizer = std::make_unique<CNN::AdamOptimizer>(0.1f);

    for (int iter = 0; iter < 1000; ++iter) {
        // 目标函数: f(x) = (x-2)²
        // 梯度: f'(x) = 2(x-2)
        CNN::Tensor grad({1});
        grad[0] = 2.0f * (x[0] - 2.0f);

        std::vector<CNN::Tensor*> params = {&x};
        std::vector<CNN::Tensor*> grads = {&grad};

        optimizer->step(params, grads);
        optimizer->zero_grad(grads);
    }

    // 验证收敛到最优解x=2
    EXPECT_NEAR(x[0], 2.0f, 1e-6f);
}
```

### 性能基准

```cpp
void benchmark_optimizers() {
    const int param_size = 1000000;
    CNN::Tensor param({param_size});
    CNN::Tensor grad({param_size});

    std::vector<std::unique_ptr<CNN::Optimizer>> optimizers;
    optimizers.push_back(std::make_unique<CNN::SGDOptimizer>(0.01f));
    optimizers.push_back(std::make_unique<CNN::AdamOptimizer>(0.001f));
    optimizers.push_back(std::make_unique<CNN::AdamWOptimizer>(0.001f));

    for (auto& opt : optimizers) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            std::vector<CNN::Tensor*> params = {&param};
            std::vector<CNN::Tensor*> grads = {&grad};
            opt->step(params, grads);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << opt->name() << " 耗时: " << duration.count() << "ms" << std::endl;
    }
}
```

## 工厂模式支持

```cpp
// 工厂函数
std::unique_ptr<Optimizer> create_optimizer(const std::string& name,
                                           const std::map<std::string, float>& config) {
    if (name == "sgd") {
        return std::make_unique<SGDOptimizer>(
            config.at("lr"),
            config.count("momentum") ? config.at("momentum") : 0.0f
        );
    } else if (name == "adam") {
        return std::make_unique<AdamOptimizer>(
            config.at("lr"),
            config.count("beta1") ? config.at("beta1") : 0.9f,
            config.count("beta2") ? config.at("beta2") : 0.999f
        );
    }
    throw std::invalid_argument("未知的优化器: " + name);
}

// 使用示例
auto optimizer = create_optimizer("adam", {
    {"lr", 0.001f},
    {"beta1", 0.9f},
    {"beta2", 0.999f}
});
```

## 未来改进方向

1. **更多优化器**

   - LAMB (Large Batch Optimization)
   - RAdam (Rectified Adam)
   - Lookahead Optimizer

2. **高级特性**

   - 分层学习率
   - 动态权重衰减
   - 梯度累积

3. **分布式支持**

   - 数据并行优化
   - 模型并行支持
   - 异步优化

4. **量化优化**
   - 混合精度优化
   - INT8 权重更新
