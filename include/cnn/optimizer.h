#pragma once

#include "tensor.h"
#include <memory>
#include <unordered_map>
#include <vector>


namespace CNN {

// 优化器基类
class Optimizer {
public:
  virtual ~Optimizer() = default;

  // 参数更新
  virtual void step(const std::vector<Tensor *> &parameters,
                    const std::vector<Tensor *> &gradients) = 0;

  // 梯度清零
  virtual void zero_grad(const std::vector<Tensor *> &gradients);

  // 学习率设置
  virtual void set_learning_rate(float lr) { learning_rate_ = lr; }
  virtual float get_learning_rate() const { return learning_rate_; }

  // 优化器名称
  virtual std::string name() const = 0;

protected:
  float learning_rate_;
};

// SGD优化器
class SGDOptimizer : public Optimizer {
public:
  SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.0f,
               float weight_decay = 0.0f);

  void step(const std::vector<Tensor *> &parameters,
            const std::vector<Tensor *> &gradients) override;
  std::string name() const override { return "SGD"; }

  void set_momentum(float momentum) { momentum_ = momentum; }
  void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }

private:
  float momentum_;
  float weight_decay_;
  std::unordered_map<void *, Tensor> velocity_buffers_;
};

// Adam优化器
class AdamOptimizer : public Optimizer {
public:
  AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                float beta2 = 0.999f, float eps = 1e-8f,
                float weight_decay = 0.0f);

  void step(const std::vector<Tensor *> &parameters,
            const std::vector<Tensor *> &gradients) override;
  std::string name() const override { return "Adam"; }

  void set_betas(float beta1, float beta2) {
    beta1_ = beta1;
    beta2_ = beta2;
  }
  void set_eps(float eps) { eps_ = eps; }
  void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }

private:
  float beta1_, beta2_;
  float eps_;
  float weight_decay_;
  int step_count_;

  std::unordered_map<void *, Tensor> m_buffers_; // 一阶矩估计
  std::unordered_map<void *, Tensor> v_buffers_; // 二阶矩估计
};

// AdamW优化器
class AdamWOptimizer : public Optimizer {
public:
  AdamWOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f,
                 float beta2 = 0.999f, float eps = 1e-8f,
                 float weight_decay = 0.01f);

  void step(const std::vector<Tensor *> &parameters,
            const std::vector<Tensor *> &gradients) override;
  std::string name() const override { return "AdamW"; }

private:
  float beta1_, beta2_;
  float eps_;
  float weight_decay_;
  int step_count_;

  std::unordered_map<void *, Tensor> m_buffers_;
  std::unordered_map<void *, Tensor> v_buffers_;
};

// RMSprop优化器
class RMSpropOptimizer : public Optimizer {
public:
  RMSpropOptimizer(float learning_rate = 0.01f, float alpha = 0.99f,
                   float eps = 1e-8f, float weight_decay = 0.0f,
                   float momentum = 0.0f);

  void step(const std::vector<Tensor *> &parameters,
            const std::vector<Tensor *> &gradients) override;
  std::string name() const override { return "RMSprop"; }

private:
  float alpha_;
  float eps_;
  float weight_decay_;
  float momentum_;

  std::unordered_map<void *, Tensor> square_avg_buffers_;
  std::unordered_map<void *, Tensor> momentum_buffers_;
};

// Adagrad优化器
class AdagradOptimizer : public Optimizer {
public:
  AdagradOptimizer(float learning_rate = 0.01f, float eps = 1e-10f,
                   float weight_decay = 0.0f);

  void step(const std::vector<Tensor *> &parameters,
            const std::vector<Tensor *> &gradients) override;
  std::string name() const override { return "Adagrad"; }

private:
  float eps_;
  float weight_decay_;
  std::unordered_map<void *, Tensor> sum_buffers_;
};

// 优化器工厂函数
std::unique_ptr<Optimizer> create_sgd_optimizer(float lr = 0.01f,
                                                float momentum = 0.0f,
                                                float weight_decay = 0.0f);
std::unique_ptr<Optimizer> create_adam_optimizer(float lr = 0.001f,
                                                 float beta1 = 0.9f,
                                                 float beta2 = 0.999f,
                                                 float eps = 1e-8f,
                                                 float weight_decay = 0.0f);
std::unique_ptr<Optimizer> create_adamw_optimizer(float lr = 0.001f,
                                                  float beta1 = 0.9f,
                                                  float beta2 = 0.999f,
                                                  float eps = 1e-8f,
                                                  float weight_decay = 0.01f);
std::unique_ptr<Optimizer> create_rmsprop_optimizer(float lr = 0.01f,
                                                    float alpha = 0.99f,
                                                    float eps = 1e-8f,
                                                    float weight_decay = 0.0f,
                                                    float momentum = 0.0f);
std::unique_ptr<Optimizer> create_adagrad_optimizer(float lr = 0.01f,
                                                    float eps = 1e-10f,
                                                    float weight_decay = 0.0f);

} // namespace CNN