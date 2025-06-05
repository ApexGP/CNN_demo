#include "cnn/optimizer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace CNN {

// Optimizer基类实现
void Optimizer::zero_grad(const std::vector<Tensor *> &gradients) {
  for (auto *grad : gradients) {
    if (grad) {
      grad->zeros();
    }
  }
}

// SGDOptimizer优化器实现
SGDOptimizer::SGDOptimizer(float learning_rate, float momentum,
                           float weight_decay)
    : momentum_(momentum), weight_decay_(weight_decay) {
  learning_rate_ = learning_rate;
}

void SGDOptimizer::step(const std::vector<Tensor *> &parameters,
                        const std::vector<Tensor *> &gradients) {
  if (parameters.size() != gradients.size()) {
    throw std::runtime_error("参数和梯度数量不匹配");
  }

  for (size_t i = 0; i < parameters.size(); ++i) {
    if (parameters[i] && gradients[i]) {
      Tensor *param = parameters[i];
      const Tensor *grad = gradients[i];

      if (momentum_ > 0.0f) {
        // 检查是否需要初始化速度缓冲区
        auto it = velocity_buffers_.find(param);
        if (it == velocity_buffers_.end()) {
          velocity_buffers_[param] = Tensor(param->shape());
          velocity_buffers_[param].zeros();
        }

        // 动量SGD: v = momentum * v + grad, param = param - lr * v
        Tensor &velocity = velocity_buffers_[param];

        for (size_t j = 0; j < param->size(); ++j) {
          velocity[j] = momentum_ * velocity[j] + (*grad)[j];
          (*param)[j] -= learning_rate_ * velocity[j];
        }
      } else {
        // 基础SGD: param = param - learning_rate * grad
        for (size_t j = 0; j < param->size(); ++j) {
          (*param)[j] -= learning_rate_ * (*grad)[j];
        }
      }
    }
  }
}

// AdamOptimizer优化器实现
AdamOptimizer::AdamOptimizer(float learning_rate, float beta1, float beta2,
                             float eps, float weight_decay)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay),
      step_count_(0) {
  learning_rate_ = learning_rate;
}

void AdamOptimizer::step(const std::vector<Tensor *> &parameters,
                         const std::vector<Tensor *> &gradients) {
  // 简化实现
  step_count_++;
  // 这里需要实现Adam算法的参数更新逻辑
}

// AdamWOptimizer实现
AdamWOptimizer::AdamWOptimizer(float learning_rate, float beta1, float beta2,
                               float eps, float weight_decay)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay),
      step_count_(0) {
  learning_rate_ = learning_rate;
}

void AdamWOptimizer::step(const std::vector<Tensor *> &parameters,
                          const std::vector<Tensor *> &gradients) {
  // 简化实现
  step_count_++;
  // 这里需要实现AdamW算法的参数更新逻辑
}

// RMSpropOptimizer实现
RMSpropOptimizer::RMSpropOptimizer(float learning_rate, float alpha, float eps,
                                   float weight_decay, float momentum)
    : alpha_(alpha), eps_(eps), weight_decay_(weight_decay),
      momentum_(momentum) {
  learning_rate_ = learning_rate;
}

void RMSpropOptimizer::step(const std::vector<Tensor *> &parameters,
                            const std::vector<Tensor *> &gradients) {
  // 简化实现
  // 这里需要实现RMSprop算法的参数更新逻辑
}

// AdagradOptimizer实现
AdagradOptimizer::AdagradOptimizer(float learning_rate, float eps,
                                   float weight_decay)
    : eps_(eps), weight_decay_(weight_decay) {
  learning_rate_ = learning_rate;
}

void AdagradOptimizer::step(const std::vector<Tensor *> &parameters,
                            const std::vector<Tensor *> &gradients) {
  // 简化实现
  // 这里需要实现Adagrad算法的参数更新逻辑
}

// 工厂函数实现
std::unique_ptr<Optimizer> create_sgd_optimizer(float lr, float momentum,
                                                float weight_decay) {
  return std::make_unique<SGDOptimizer>(lr, momentum, weight_decay);
}

std::unique_ptr<Optimizer> create_adam_optimizer(float lr, float beta1,
                                                 float beta2, float eps,
                                                 float weight_decay) {
  return std::make_unique<AdamOptimizer>(lr, beta1, beta2, eps, weight_decay);
}

std::unique_ptr<Optimizer> create_adamw_optimizer(float lr, float beta1,
                                                  float beta2, float eps,
                                                  float weight_decay) {
  return std::make_unique<AdamWOptimizer>(lr, beta1, beta2, eps, weight_decay);
}

std::unique_ptr<Optimizer> create_rmsprop_optimizer(float lr, float alpha,
                                                    float eps,
                                                    float weight_decay,
                                                    float momentum) {
  return std::make_unique<RMSpropOptimizer>(lr, alpha, eps, weight_decay,
                                            momentum);
}

std::unique_ptr<Optimizer> create_adagrad_optimizer(float lr, float eps,
                                                    float weight_decay) {
  return std::make_unique<AdagradOptimizer>(lr, eps, weight_decay);
}

} // namespace CNN