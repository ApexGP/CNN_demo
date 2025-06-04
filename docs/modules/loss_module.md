# 损失函数模块设计文档

## 概述

损失函数模块实现了深度学习中常用的损失函数，用于衡量模型预测与真实标签之间的差异，并提供反向传播所需的梯度信息。

## 设计理念

### 1. 统一接口设计

```
        LossFunction (抽象基类)
               │
    ┌──────────┼──────────┐
    │          │          │
MSELoss   CrossEntropyLoss  HuberLoss
    │          │          │
    │          │      L1Loss
    │          │          │
FocalLoss  BinaryCrossEntropyLoss  │
    │          │          │
KLDivLoss      │    CosineEmbeddingLoss
```

### 2. 核心设计原则

- **统一接口**: 所有损失函数继承自`LossFunction`基类
- **前向后向分离**: 明确的损失计算和梯度计算接口
- **reduction 支持**: 支持 sum、mean、none 等聚合方式
- **数值稳定性**: 处理数值计算中的稳定性问题
- **灵活配置**: 支持各种超参数配置

## 模块结构

### 基类 `LossFunction`

**文件位置**: `include/cnn/loss.h`

```cpp
namespace CNN {
class LossFunction {
public:
    virtual ~LossFunction() = default;

    // 核心接口
    virtual float forward(const Tensor& predictions, const Tensor& targets) = 0;
    virtual Tensor backward(const Tensor& predictions, const Tensor& targets) = 0;

    // 配置接口
    virtual void set_reduction(const std::string& reduction) { reduction_ = reduction; }
    virtual std::string get_reduction() const { return reduction_; }

    // 信息接口
    virtual std::string name() const = 0;

protected:
    std::string reduction_ = "mean";  // sum, mean, none

    // 辅助函数
    float apply_reduction(const Tensor& losses) const;
};
}
```

## 具体损失函数实现

### 1. 均方误差损失 (MSELoss)

**数学公式**:

```
MSE = (1/n) * Σ(y_pred - y_true)²
```

**用途**: 回归任务的标准损失函数

**实现**:

```cpp
class MSELoss : public LossFunction {
public:
    float forward(const Tensor& predictions, const Tensor& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::invalid_argument("预测和目标形状不匹配");
        }

        // 计算均方误差
        float sum = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i] - targets[i];
            sum += diff * diff;
        }

        if (reduction_ == "mean") {
            return sum / predictions.size();
        } else if (reduction_ == "sum") {
            return sum;
        } else {
            return sum; // "none" - 简化处理
        }
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::invalid_argument("预测和目标形状不匹配");
        }

        // MSE的梯度是 2*(predictions - targets) / n
        Tensor grad(predictions.shape());
        float scale = 2.0f;
        if (reduction_ == "mean") {
            scale /= predictions.size();
        }

        for (size_t i = 0; i < predictions.size(); ++i) {
            grad[i] = scale * (predictions[i] - targets[i]);
        }

        return grad;
    }

    std::string name() const override { return "MSELoss"; }
};
```

### 2. 交叉熵损失 (CrossEntropyLoss)

**数学公式**:

```
CrossEntropy = -Σ y_true * log(softmax(y_pred))
```

**用途**: 多分类任务的标准损失函数

**实现要点**:

- 数值稳定的 softmax 计算
- 支持从 logits 和概率两种输入
- 可选的标签平滑

**实现**:

```cpp
class CrossEntropyLoss : public LossFunction {
private:
    bool from_logits_;
    float label_smoothing_;

public:
    CrossEntropyLoss(bool from_logits = true, float label_smoothing = 0.0f)
        : from_logits_(from_logits), label_smoothing_(label_smoothing) {}

    float forward(const Tensor& predictions, const Tensor& targets) override {
        const auto& pred_shape = predictions.shape();

        if (pred_shape.size() != 2) {
            throw std::invalid_argument("预测张量必须是2D [batch_size, num_classes]");
        }

        int batch_size = pred_shape[0];
        int num_classes = pred_shape[1];
        float loss = 0.0f;

        // 简化实现：假设targets是类别索引
        for (int i = 0; i < batch_size; ++i) {
            int target_class = static_cast<int>(targets[i]);

            if (target_class < 0 || target_class >= num_classes) {
                throw std::out_of_range("目标类别索引超出范围");
            }

            if (from_logits_) {
                // 数值稳定的softmax计算
                float max_val = predictions[i * num_classes];
                for (int j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, predictions[i * num_classes + j]);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < num_classes; ++j) {
                    sum_exp += std::exp(predictions[i * num_classes + j] - max_val);
                }

                float log_prob = predictions[i * num_classes + target_class] - max_val - std::log(sum_exp);
                loss -= log_prob;
            } else {
                // 直接使用概率
                float p = std::max(predictions[i * num_classes + target_class], 1e-7f);
                loss -= std::log(p);
            }
        }

        return apply_reduction(Tensor({loss}));
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        const auto& pred_shape = predictions.shape();
        int batch_size = pred_shape[0];
        int num_classes = pred_shape[1];

        Tensor grad(pred_shape);
        grad.zeros();

        for (int i = 0; i < batch_size; ++i) {
            int target_class = static_cast<int>(targets[i]);

            if (from_logits_) {
                // 计算softmax概率
                float max_val = predictions[i * num_classes];
                for (int j = 1; j < num_classes; ++j) {
                    max_val = std::max(max_val, predictions[i * num_classes + j]);
                }

                float sum_exp = 0.0f;
                for (int j = 0; j < num_classes; ++j) {
                    grad[i * num_classes + j] = std::exp(predictions[i * num_classes + j] - max_val);
                    sum_exp += grad[i * num_classes + j];
                }

                for (int j = 0; j < num_classes; ++j) {
                    grad[i * num_classes + j] /= sum_exp;
                }

                grad[i * num_classes + target_class] -= 1.0f;
            } else {
                // 直接计算梯度
                for (int j = 0; j < num_classes; ++j) {
                    grad[i * num_classes + j] = predictions[i * num_classes + j];
                }
                grad[i * num_classes + target_class] -= 1.0f;
            }
        }

        if (reduction_ == "mean") {
            float scale = 1.0f / batch_size;
            for (size_t i = 0; i < grad.size(); ++i) {
                grad[i] *= scale;
            }
        }

        return grad;
    }

    std::string name() const override { return "CrossEntropyLoss"; }
};
```

### 3. 二元交叉熵损失 (BinaryCrossEntropyLoss)

**数学公式**:

```
BCE = -(y_true * log(p) + (1-y_true) * log(1-p))
```

**用途**: 二分类和多标签分类任务

**特性**:

- 支持正样本权重调整
- 数值稳定的实现
- 支持从 logits 输入

**实现**:

```cpp
class BinaryCrossEntropyLoss : public LossFunction {
private:
    bool from_logits_;
    float pos_weight_;

public:
    BinaryCrossEntropyLoss(bool from_logits = false, float pos_weight = 1.0f)
        : from_logits_(from_logits), pos_weight_(pos_weight) {}

    float forward(const Tensor& predictions, const Tensor& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::invalid_argument("预测和目标形状不匹配");
        }

        float loss = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float p = predictions[i];
            float t = targets[i];

            if (from_logits_) {
                // 使用log-sum-exp trick避免数值不稳定
                float max_val = std::max(0.0f, p);
                loss += max_val - t * p + std::log(1.0f + std::exp(-std::abs(p)));
            } else {
                p = std::max(std::min(p, 1.0f - 1e-7f), 1e-7f); // 数值稳定性
                loss -= t * std::log(p) + (1.0f - t) * std::log(1.0f - p);
            }

            if (t > 0.5f) { // 正样本
                loss *= pos_weight_;
            }
        }

        return apply_reduction(Tensor({loss}));
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());

        for (size_t i = 0; i < predictions.size(); ++i) {
            float p = predictions[i];
            float t = targets[i];

            if (from_logits_) {
                float sigmoid_p = 1.0f / (1.0f + std::exp(-p));
                grad[i] = sigmoid_p - t;
            } else {
                p = std::max(std::min(p, 1.0f - 1e-7f), 1e-7f);
                grad[i] = (p - t) / (p * (1.0f - p));
            }

            if (t > 0.5f) { // 正样本
                grad[i] *= pos_weight_;
            }
        }

        if (reduction_ == "mean") {
            float scale = 1.0f / predictions.size();
            for (size_t i = 0; i < grad.size(); ++i) {
                grad[i] *= scale;
            }
        }

        return grad;
    }

    std::string name() const override { return "BinaryCrossEntropyLoss"; }
};
```

### 4. Huber 损失 (HuberLoss)

**数学公式**:

```
Huber(x) = {
    0.5 * x²,           if |x| <= δ
    δ * (|x| - 0.5*δ),  if |x| > δ
}
```

**用途**: 对异常值较为鲁棒的回归损失

**特性**:

- 结合了 MSE 和 MAE 的优点
- 可配置的 δ 参数
- 对异常值不敏感

**实现**:

```cpp
class HuberLoss : public LossFunction {
private:
    float delta_;

public:
    HuberLoss(float delta = 1.0f) : delta_(delta) {}

    float forward(const Tensor& predictions, const Tensor& targets) override {
        if (predictions.shape() != targets.shape()) {
            throw std::invalid_argument("预测和目标形状不匹配");
        }

        float loss = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = std::abs(predictions[i] - targets[i]);
            if (diff <= delta_) {
                loss += 0.5f * diff * diff;
            } else {
                loss += delta_ * (diff - 0.5f * delta_);
            }
        }

        return apply_reduction(Tensor({loss}));
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        Tensor grad(predictions.shape());

        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i] - targets[i];
            if (std::abs(diff) <= delta_) {
                grad[i] = diff;
            } else {
                grad[i] = delta_ * (diff > 0 ? 1.0f : -1.0f);
            }
        }

        if (reduction_ == "mean") {
            float scale = 1.0f / predictions.size();
            for (size_t i = 0; i < grad.size(); ++i) {
                grad[i] *= scale;
            }
        }

        return grad;
    }

    std::string name() const override { return "HuberLoss"; }
};
```

### 5. Focal Loss

**数学公式**:

```
FocalLoss = -α * (1-p)^γ * log(p)
```

**用途**: 解决类别不平衡问题

**特性**:

- 关注难分类样本
- 可配置的 α 和 γ 参数
- 适用于目标检测等任务

**实现**:

```cpp
class FocalLoss : public LossFunction {
private:
    float alpha_, gamma_;
    bool from_logits_;

public:
    FocalLoss(float alpha = 1.0f, float gamma = 2.0f, bool from_logits = true)
        : alpha_(alpha), gamma_(gamma), from_logits_(from_logits) {}

    float forward(const Tensor& predictions, const Tensor& targets) override {
        // 实现细节（简化）
        // 需要根据具体需求实现多分类或二分类版本
        return 0.0f;
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        // 实现细节（简化）
        return predictions.clone();
    }

    std::string name() const override { return "FocalLoss"; }
};
```

## 损失函数特性对比

| 损失函数                   | 适用任务   | 对异常值 | 数值稳定性 | 计算复杂度 |
| -------------------------- | ---------- | -------- | ---------- | ---------- |
| **MSELoss**                | 回归       | 敏感     | 良好       | 低         |
| **CrossEntropyLoss**       | 多分类     | 中等     | 需要注意   | 中等       |
| **BinaryCrossEntropyLoss** | 二分类     | 中等     | 需要注意   | 低         |
| **HuberLoss**              | 回归       | 鲁棒     | 良好       | 低         |
| **L1Loss**                 | 回归       | 鲁棒     | 良好       | 低         |
| **FocalLoss**              | 不平衡分类 | 中等     | 需要注意   | 高         |

## 数值稳定性处理

### 1. Softmax 稳定性

```cpp
// 数值稳定的softmax实现
Tensor stable_softmax(const Tensor& logits) {
    Tensor result(logits.shape());

    // 减去最大值防止溢出
    float max_val = *std::max_element(logits.data(), logits.data() + logits.size());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }

    // 归一化
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] /= sum;
    }

    return result;
}
```

### 2. 对数概率稳定性

```cpp
// 数值稳定的log-sum-exp
float log_sum_exp(const std::vector<float>& values) {
    float max_val = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;

    for (float v : values) {
        sum += std::exp(v - max_val);
    }

    return max_val + std::log(sum);
}
```

## 依赖关系

### 内部依赖

```
LossFunction接口
    ↓
各具体损失函数实现
    ↓
Tensor运算 (加法、乘法、对数等)
    ↓
math_core (数学函数)
    ↓
C++标准库 (cmath)
```

### 无第三方依赖

损失函数模块主要依赖：

- **C++标准库**: `<cmath>`, `<algorithm>`, `<stdexcept>`
- **Tensor 模块**: 用于数据操作
- **无外部依赖**: 纯 C++实现

## 使用示例

### 基础使用

```cpp
#include "cnn/loss.h"

// 回归任务
auto mse_loss = std::make_unique<CNN::MSELoss>();
CNN::Tensor predictions({3}); // [1.0, 2.0, 3.0]
CNN::Tensor targets({3});     // [1.5, 2.2, 2.8]

float loss_value = mse_loss->forward(predictions, targets);
CNN::Tensor gradients = mse_loss->backward(predictions, targets);

// 分类任务
auto ce_loss = std::make_unique<CNN::CrossEntropyLoss>(true);
CNN::Tensor logits({2, 3});   // batch_size=2, num_classes=3
CNN::Tensor labels({2});      // [0, 2] 类别标签

float ce_loss_value = ce_loss->forward(logits, labels);
CNN::Tensor ce_gradients = ce_loss->backward(logits, labels);
```

### 高级配置

```cpp
// 带权重的二元交叉熵
auto bce_loss = std::make_unique<CNN::BinaryCrossEntropyLoss>(
    true,   // from_logits
    2.0f    // pos_weight，正样本权重
);

// 设置reduction方式
bce_loss->set_reduction("sum");

// Huber损失配置
auto huber_loss = std::make_unique<CNN::HuberLoss>(1.5f); // delta=1.5

// Focal Loss配置
auto focal_loss = std::make_unique<CNN::FocalLoss>(
    0.25f,  // alpha
    2.0f,   // gamma
    true    // from_logits
);
```

### 自定义损失函数

```cpp
class CustomLoss : public CNN::LossFunction {
public:
    float forward(const Tensor& predictions, const Tensor& targets) override {
        // 自定义损失计算
        // 例如：加权MSE
        float loss = 0.0f;
        for (size_t i = 0; i < predictions.size(); ++i) {
            float diff = predictions[i] - targets[i];
            float weight = get_sample_weight(i);
            loss += weight * diff * diff;
        }
        return loss / predictions.size();
    }

    Tensor backward(const Tensor& predictions, const Tensor& targets) override {
        // 自定义梯度计算
        Tensor grad(predictions.shape());
        for (size_t i = 0; i < grad.size(); ++i) {
            float weight = get_sample_weight(i);
            grad[i] = 2.0f * weight * (predictions[i] - targets[i]) / predictions.size();
        }
        return grad;
    }

    std::string name() const override { return "CustomLoss"; }

private:
    float get_sample_weight(size_t index) {
        // 根据样本索引返回权重
        return 1.0f; // 简化实现
    }
};
```

## 性能优化

### 1. 向量化计算

```cpp
// 使用OpenMP并行化
void parallel_mse_forward(const Tensor& pred, const Tensor& target, float& loss) {
    float sum = 0.0f;

    #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < pred.size(); ++i) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }

    loss = sum / pred.size();
}
```

### 2. 内存优化

```cpp
// 原地梯度计算
void inplace_gradient_computation(Tensor& grad, const Tensor& pred, const Tensor& target) {
    // 复用现有内存，避免额外分配
    const float scale = 2.0f / pred.size();

    #pragma omp parallel for
    for (size_t i = 0; i < grad.size(); ++i) {
        grad[i] = scale * (pred[i] - target[i]);
    }
}
```

## 测试与验证

### 梯度检查

```cpp
bool gradient_check_loss(LossFunction* loss, const Tensor& pred, const Tensor& target) {
    const float eps = 1e-5f;
    const float tolerance = 1e-3f;

    // 解析梯度
    Tensor analytical_grad = loss->backward(pred, target);

    // 数值梯度
    Tensor numerical_grad(pred.shape());
    for (size_t i = 0; i < pred.size(); ++i) {
        Tensor pred_plus = pred.clone();
        Tensor pred_minus = pred.clone();

        pred_plus[i] += eps;
        pred_minus[i] -= eps;

        float loss_plus = loss->forward(pred_plus, target);
        float loss_minus = loss->forward(pred_minus, target);

        numerical_grad[i] = (loss_plus - loss_minus) / (2.0f * eps);
    }

    // 比较差异
    for (size_t i = 0; i < analytical_grad.size(); ++i) {
        float diff = std::abs(analytical_grad[i] - numerical_grad[i]);
        if (diff > tolerance) {
            return false;
        }
    }

    return true;
}
```

### 性能基准

```cpp
void benchmark_loss_functions() {
    const size_t data_size = 1000000;
    CNN::Tensor pred({data_size});
    CNN::Tensor target({data_size});

    // 初始化数据
    pred.rand(0.0f, 1.0f);
    target.rand(0.0f, 1.0f);

    std::vector<std::unique_ptr<CNN::LossFunction>> losses;
    losses.push_back(std::make_unique<CNN::MSELoss>());
    losses.push_back(std::make_unique<CNN::HuberLoss>());

    for (auto& loss : losses) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < 1000; ++i) {
            float loss_val = loss->forward(pred, target);
            CNN::Tensor grad = loss->backward(pred, target);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << loss->name() << " 耗时: " << duration.count() << "ms" << std::endl;
    }
}
```

## 工厂模式支持

```cpp
// 损失函数工厂
std::unique_ptr<LossFunction> create_loss_function(const std::string& name,
                                                  const std::map<std::string, float>& config) {
    if (name == "mse") {
        return std::make_unique<MSELoss>();
    } else if (name == "cross_entropy") {
        bool from_logits = config.count("from_logits") ? config.at("from_logits") > 0.5f : true;
        return std::make_unique<CrossEntropyLoss>(from_logits);
    } else if (name == "huber") {
        float delta = config.count("delta") ? config.at("delta") : 1.0f;
        return std::make_unique<HuberLoss>(delta);
    }
    throw std::invalid_argument("未知的损失函数: " + name);
}

// 使用示例
auto loss = create_loss_function("huber", {{"delta", 1.5f}});
```

## 未来改进方向

1. **更多损失函数**

   - Dice Loss (用于分割任务)
   - Triplet Loss (用于度量学习)
   - Contrastive Loss (用于孪生网络)

2. **高级特性**

   - 标签平滑 (Label Smoothing)
   - 混合损失函数
   - 自适应权重调整

3. **优化技术**

   - GPU 加速实现
   - 混合精度计算
   - 内存池管理

4. **应用特化**
   - 时序预测损失
   - 图像分割损失
   - 强化学习损失函数
