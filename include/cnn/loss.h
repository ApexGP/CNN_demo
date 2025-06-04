#pragma once

#include "tensor.h"
#include <memory>
#include <string>

namespace CNN {

// 损失函数基类
class LossFunction {
public:
  virtual ~LossFunction() = default;

  // 计算损失
  virtual float forward(const Tensor &predictions, const Tensor &targets) = 0;

  // 计算梯度
  virtual Tensor backward(const Tensor &predictions, const Tensor &targets) = 0;

  // 损失函数名称
  virtual std::string name() const = 0;

  // 实用方法
  virtual void set_reduction(const std::string &reduction) {
    reduction_ = reduction;
  }
  virtual std::string get_reduction() const { return reduction_; }

protected:
  std::string reduction_ = "mean"; // "mean", "sum", "none"

  // 辅助函数
  float apply_reduction(const Tensor &losses) const;
};

// 均方误差损失
class MSELoss : public LossFunction {
public:
  MSELoss() = default;

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "MSELoss"; }
};

// 交叉熵损失
class CrossEntropyLoss : public LossFunction {
public:
  CrossEntropyLoss(bool from_logits = true, float label_smoothing = 0.0f);

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "CrossEntropyLoss"; }

  void set_from_logits(bool from_logits) { from_logits_ = from_logits; }
  void set_label_smoothing(float label_smoothing) {
    label_smoothing_ = label_smoothing;
  }

private:
  bool from_logits_;
  float label_smoothing_;

  Tensor apply_label_smoothing(const Tensor &targets) const;
};

// 二元交叉熵损失
class BinaryCrossEntropyLoss : public LossFunction {
public:
  BinaryCrossEntropyLoss(bool from_logits = false, float pos_weight = 1.0f);

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "BinaryCrossEntropyLoss"; }

  void set_from_logits(bool from_logits) { from_logits_ = from_logits; }
  void set_pos_weight(float pos_weight) { pos_weight_ = pos_weight; }

private:
  bool from_logits_;
  float pos_weight_;
};

// Huber损失（平滑L1损失）
class HuberLoss : public LossFunction {
public:
  HuberLoss(float delta = 1.0f) : delta_(delta) {}

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "HuberLoss"; }

  void set_delta(float delta) { delta_ = delta; }

private:
  float delta_;
};

// L1损失（平均绝对误差）
class L1Loss : public LossFunction {
public:
  L1Loss() = default;

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "L1Loss"; }
};

// Focal Loss（用于处理类别不平衡）
class FocalLoss : public LossFunction {
public:
  FocalLoss(float alpha = 1.0f, float gamma = 2.0f, bool from_logits = true);

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "FocalLoss"; }

  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_gamma(float gamma) { gamma_ = gamma; }

private:
  float alpha_;
  float gamma_;
  bool from_logits_;
};

// KL散度损失
class KLDivLoss : public LossFunction {
public:
  KLDivLoss(bool log_target = false) : log_target_(log_target) {}

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "KLDivLoss"; }

  void set_log_target(bool log_target) { log_target_ = log_target; }

private:
  bool log_target_;
};

// 余弦嵌入损失
class CosineEmbeddingLoss : public LossFunction {
public:
  CosineEmbeddingLoss(float margin = 0.0f) : margin_(margin) {}

  float forward(const Tensor &predictions, const Tensor &targets) override;
  Tensor backward(const Tensor &predictions, const Tensor &targets) override;

  std::string name() const override { return "CosineEmbeddingLoss"; }

  void set_margin(float margin) { margin_ = margin; }

private:
  float margin_;
};

// 损失函数工厂函数
std::unique_ptr<LossFunction> create_mse_loss();
std::unique_ptr<LossFunction>
create_cross_entropy_loss(bool from_logits = true,
                          float label_smoothing = 0.0f);
std::unique_ptr<LossFunction>
create_binary_cross_entropy_loss(bool from_logits = false,
                                 float pos_weight = 1.0f);
std::unique_ptr<LossFunction> create_huber_loss(float delta = 1.0f);
std::unique_ptr<LossFunction> create_l1_loss();
std::unique_ptr<LossFunction> create_focal_loss(float alpha = 1.0f,
                                                float gamma = 2.0f,
                                                bool from_logits = true);
std::unique_ptr<LossFunction> create_kl_div_loss(bool log_target = false);
std::unique_ptr<LossFunction> create_cosine_embedding_loss(float margin = 0.0f);

} // namespace CNN