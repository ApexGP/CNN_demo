#include "cnn/loss.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace CNN {

// LossFunction基类实现
float LossFunction::apply_reduction(const Tensor &losses) const {
  if (losses.size() == 0) {
    return 0.0f;
  }

  const float *data = losses.data();
  if (reduction_ == "sum") {
    float sum = 0.0f;
    for (size_t i = 0; i < losses.size(); ++i) {
      sum += data[i];
    }
    return sum;
  } else if (reduction_ == "mean") {
    float sum = 0.0f;
    for (size_t i = 0; i < losses.size(); ++i) {
      sum += data[i];
    }
    return sum / losses.size();
  } else {          // "none"
    return data[0]; // 返回第一个元素
  }
}

// MSE损失实现
float MSELoss::forward(const Tensor &predictions, const Tensor &targets) {
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

Tensor MSELoss::backward(const Tensor &predictions, const Tensor &targets) {
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

// 交叉熵损失实现
CrossEntropyLoss::CrossEntropyLoss(bool from_logits, float label_smoothing)
    : from_logits_(from_logits), label_smoothing_(label_smoothing) {}

float CrossEntropyLoss::forward(const Tensor &predictions,
                                const Tensor &targets) {
  const auto &pred_shape = predictions.shape();
  const auto &target_shape = targets.shape();

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

    // 如果是logits，需要应用softmax
    if (from_logits_) {
      // 简化的softmax计算
      float max_val = predictions[i * num_classes];
      for (int j = 1; j < num_classes; ++j) {
        max_val = std::max(max_val, predictions[i * num_classes + j]);
      }

      float sum_exp = 0.0f;
      for (int j = 0; j < num_classes; ++j) {
        sum_exp += std::exp(predictions[i * num_classes + j] - max_val);
      }

      float log_prob = predictions[i * num_classes + target_class] - max_val -
                       std::log(sum_exp);
      loss -= log_prob;
    } else {
      // 直接使用概率
      float p = std::max(predictions[i * num_classes + target_class], 1e-7f);
      loss -= std::log(p);
    }
  }

  if (reduction_ == "mean") {
    return loss / batch_size;
  } else if (reduction_ == "sum") {
    return loss;
  } else {
    return loss; // "none" - 简化处理
  }
}

Tensor CrossEntropyLoss::backward(const Tensor &predictions,
                                  const Tensor &targets) {
  const auto &pred_shape = predictions.shape();

  if (pred_shape.size() != 2) {
    throw std::invalid_argument("预测张量必须是2D [batch_size, num_classes]");
  }

  int batch_size = pred_shape[0];
  int num_classes = pred_shape[1];

  Tensor grad(pred_shape);
  grad.zeros();

  // 简化实现
  for (int i = 0; i < batch_size; ++i) {
    int target_class = static_cast<int>(targets[i]);

    if (from_logits_) {
      // 计算softmax
      float max_val = predictions[i * num_classes];
      for (int j = 1; j < num_classes; ++j) {
        max_val = std::max(max_val, predictions[i * num_classes + j]);
      }

      float sum_exp = 0.0f;
      for (int j = 0; j < num_classes; ++j) {
        grad[i * num_classes + j] =
            std::exp(predictions[i * num_classes + j] - max_val);
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

Tensor CrossEntropyLoss::apply_label_smoothing(const Tensor &targets) const {
  // 简化实现 - 直接返回原始targets
  return targets.clone();
}

// 二元交叉熵损失实现
BinaryCrossEntropyLoss::BinaryCrossEntropyLoss(bool from_logits,
                                               float pos_weight)
    : from_logits_(from_logits), pos_weight_(pos_weight) {}

float BinaryCrossEntropyLoss::forward(const Tensor &predictions,
                                      const Tensor &targets) {
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

  return apply_reduction(Tensor({static_cast<size_t>(1)}).fill(loss));
}

Tensor BinaryCrossEntropyLoss::backward(const Tensor &predictions,
                                        const Tensor &targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument("预测和目标形状不匹配");
  }

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

// Huber损失实现
float HuberLoss::forward(const Tensor &predictions, const Tensor &targets) {
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

  return apply_reduction(Tensor({static_cast<size_t>(1)}).fill(loss));
}

Tensor HuberLoss::backward(const Tensor &predictions, const Tensor &targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument("预测和目标形状不匹配");
  }

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

// L1损失实现
float L1Loss::forward(const Tensor &predictions, const Tensor &targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument("预测和目标形状不匹配");
  }

  float loss = 0.0f;
  for (size_t i = 0; i < predictions.size(); ++i) {
    loss += std::abs(predictions[i] - targets[i]);
  }

  return apply_reduction(Tensor({static_cast<size_t>(1)}).fill(loss));
}

Tensor L1Loss::backward(const Tensor &predictions, const Tensor &targets) {
  if (predictions.shape() != targets.shape()) {
    throw std::invalid_argument("预测和目标形状不匹配");
  }

  Tensor grad(predictions.shape());

  for (size_t i = 0; i < predictions.size(); ++i) {
    float diff = predictions[i] - targets[i];
    grad[i] = (diff > 0) ? 1.0f : -1.0f;
  }

  if (reduction_ == "mean") {
    float scale = 1.0f / predictions.size();
    for (size_t i = 0; i < grad.size(); ++i) {
      grad[i] *= scale;
    }
  }

  return grad;
}

// Focal Loss实现
FocalLoss::FocalLoss(float alpha, float gamma, bool from_logits)
    : alpha_(alpha), gamma_(gamma), from_logits_(from_logits) {}

float FocalLoss::forward(const Tensor &predictions, const Tensor &targets) {
  // 简化实现
  return 0.0f;
}

Tensor FocalLoss::backward(const Tensor &predictions, const Tensor &targets) {
  // 简化实现
  return predictions.clone();
}

// KL散度损失实现
float KLDivLoss::forward(const Tensor &predictions, const Tensor &targets) {
  // 简化实现
  return 0.0f;
}

Tensor KLDivLoss::backward(const Tensor &predictions, const Tensor &targets) {
  // 简化实现
  return predictions.clone();
}

// 余弦嵌入损失实现
float CosineEmbeddingLoss::forward(const Tensor &predictions,
                                   const Tensor &targets) {
  // 简化实现
  return 0.0f;
}

Tensor CosineEmbeddingLoss::backward(const Tensor &predictions,
                                     const Tensor &targets) {
  // 简化实现
  return predictions.clone();
}

// 工厂函数实现
std::unique_ptr<LossFunction> create_mse_loss() {
  return std::make_unique<MSELoss>();
}

std::unique_ptr<LossFunction> create_cross_entropy_loss(bool from_logits,
                                                        float label_smoothing) {
  return std::make_unique<CrossEntropyLoss>(from_logits, label_smoothing);
}

std::unique_ptr<LossFunction>
create_binary_cross_entropy_loss(bool from_logits, float pos_weight) {
  return std::make_unique<BinaryCrossEntropyLoss>(from_logits, pos_weight);
}

std::unique_ptr<LossFunction> create_huber_loss(float delta) {
  return std::make_unique<HuberLoss>(delta);
}

std::unique_ptr<LossFunction> create_l1_loss() {
  return std::make_unique<L1Loss>();
}

std::unique_ptr<LossFunction> create_focal_loss(float alpha, float gamma,
                                                bool from_logits) {
  return std::make_unique<FocalLoss>(alpha, gamma, from_logits);
}

std::unique_ptr<LossFunction> create_kl_div_loss(bool log_target) {
  return std::make_unique<KLDivLoss>(log_target);
}

std::unique_ptr<LossFunction> create_cosine_embedding_loss(float margin) {
  return std::make_unique<CosineEmbeddingLoss>(margin);
}

} // namespace CNN