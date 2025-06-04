/**
 * @file network.h
 * @brief C++封装的神经网络类定义
 */

#ifndef CNN_NETWORK_H_
#define CNN_NETWORK_H_

#include "layers.h"
#include "loss.h"
#include "optimizer.h"
#include "tensor.h"
#include "utils.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace CNN {

struct TrainingMetrics {
  std::vector<float> train_losses;
  std::vector<float> train_accuracies;
  std::vector<float> val_losses;
  std::vector<float> val_accuracies;
  std::vector<float> learning_rates;
  int total_epochs;
  float best_val_accuracy;
  int best_epoch;
};

// 定义优化器类型
enum OptimizerType {
  OPTIMIZER_SGD,
  OPTIMIZER_MOMENTUM,
  OPTIMIZER_ADAGRAD,
  OPTIMIZER_RMSPROP,
  OPTIMIZER_ADAM
};

// 定义损失函数类型
enum LossType { LOSS_MSE, LOSS_CROSS_ENTROPY, LOSS_BINARY_CROSS_ENTROPY };

class Network {
public:
  Network();
  ~Network();

  // 网络构建
  template <typename LayerType, typename... Args>
  void add_layer(Args &&...args) {
    auto layer = std::make_unique<LayerType>(std::forward<Args>(args)...);
    layers_.push_back(std::move(layer));
  }

  void add_layer(std::unique_ptr<Layer> layer);
  void clear_layers();

  // 便捷的层添加方法
  void add_conv_layer(int out_channels, int kernel_size, int stride = 1,
                      int padding = 0, bool bias = true);
  void add_fc_layer(int out_features, bool bias = true);
  void add_relu_layer();
  void add_sigmoid_layer();
  void add_tanh_layer();
  void add_softmax_layer(int dim = -1);
  void add_maxpool_layer(int kernel_size, int stride = -1, int padding = 0);
  void add_avgpool_layer(int kernel_size, int stride = -1, int padding = 0);
  void add_dropout_layer(float p = 0.5);
  void add_batchnorm_layer(int num_features, float eps = 1e-5,
                           float momentum = 0.1);
  void add_flatten_layer();

  // 前向传播
  Tensor forward(const Tensor &input);
  Tensor predict(const Tensor &input);
  std::vector<Tensor> predict_batch(const std::vector<Tensor> &inputs);

  // 训练
  void train(const std::vector<Tensor> &train_data,
             const std::vector<Tensor> &train_labels, int epochs = 100,
             int batch_size = 32, float learning_rate = 0.001f);

  void train_with_validation(const std::vector<Tensor> &train_data,
                             const std::vector<Tensor> &train_labels,
                             const std::vector<Tensor> &val_data,
                             const std::vector<Tensor> &val_labels,
                             int epochs = 100, int batch_size = 32,
                             float learning_rate = 0.001f);

  // 评估
  float evaluate(const std::vector<Tensor> &test_data,
                 const std::vector<Tensor> &test_labels);
  float calculate_accuracy(const std::vector<Tensor> &data,
                           const std::vector<Tensor> &labels);

  // 优化器和损失函数设置
  void set_optimizer(std::unique_ptr<Optimizer> optimizer);
  void set_loss_function(std::unique_ptr<LossFunction> loss_fn);

  // 模式设置
  void set_training_mode(bool training);
  void train_mode() { set_training_mode(true); }
  void eval_mode() { set_training_mode(false); }

  // 设备管理
  void to_cpu();
  void to_gpu();
  Device get_device() const { return device_; }

  // 网络信息
  void print_summary(const std::vector<int> &input_shape) const;
  int get_num_parameters() const;
  std::vector<std::vector<int>>
  get_layer_output_shapes(const std::vector<int> &input_shape) const;

  // 模型保存和加载
  void save_model(const std::string &filename) const;
  void load_model(const std::string &filename);
  void save_weights(const std::string &filename) const;
  void load_weights(const std::string &filename);

  // 训练监控
  const TrainingMetrics &get_training_metrics() const { return metrics_; }
  void reset_metrics();

  // 可视化支持
  void visualize_training(const std::string &save_path = "") const;
  void visualize_network_architecture(const std::vector<int> &input_shape,
                                      const std::string &save_path = "") const;
  void visualize_feature_maps(const Tensor &input, int layer_index,
                              const std::string &save_path = "") const;
  void visualize_filters(int layer_index,
                         const std::string &save_path = "") const;

  // 学习率调度
  void
  set_learning_rate_scheduler(const std::string &scheduler_type,
                              const std::map<std::string, float> &params = {});
  void update_learning_rate(int epoch);

  // 早停机制
  void enable_early_stopping(int patience = 10, float min_delta = 1e-4);
  void disable_early_stopping();

  // 正则化
  void set_weight_decay(float weight_decay) { weight_decay_ = weight_decay; }
  void set_gradient_clipping(float max_norm) { gradient_clip_norm_ = max_norm; }

  // 数据增强支持
  void enable_data_augmentation(bool enable = true) {
    data_augmentation_ = enable;
  }

  // 调试功能
  void enable_debug_mode(bool enable = true) { debug_mode_ = enable; }
  void print_gradients() const;
  void check_gradients(const Tensor &input, const Tensor &target) const;

private:
  std::vector<std::unique_ptr<Layer>> layers_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<LossFunction> loss_function_;

  Device device_;
  bool training_mode_;

  // 训练相关参数
  TrainingMetrics metrics_;
  float weight_decay_;
  float gradient_clip_norm_;
  bool data_augmentation_;
  bool debug_mode_;

  // 学习率调度
  std::string lr_scheduler_type_;
  std::map<std::string, float> lr_scheduler_params_;
  float initial_learning_rate_;

  // 早停机制
  bool early_stopping_enabled_;
  int early_stopping_patience_;
  float early_stopping_min_delta_;
  int patience_counter_;
  float best_val_loss_;

  // 内部方法
  void backward(const Tensor &loss_grad);
  void update_parameters();
  void apply_weight_decay();
  void clip_gradients();
  std::vector<std::vector<Tensor>>
  create_batches(const std::vector<Tensor> &data,
                 const std::vector<Tensor> &labels, int batch_size);
  Tensor augment_data(const Tensor &input) const;
  bool check_early_stopping(float val_loss);
  void log_training_progress(int epoch, int total_epochs, float train_loss,
                             float train_acc, float val_loss = -1,
                             float val_acc = -1) const;

  // 可视化辅助函数
  void save_plot_data(const std::vector<float> &x_data,
                      const std::vector<float> &y_data,
                      const std::string &filename, const std::string &title,
                      const std::string &xlabel,
                      const std::string &ylabel) const;
};

// 网络构建辅助函数
std::unique_ptr<Network> create_lenet5(int num_classes = 10);
std::unique_ptr<Network> create_alexnet(int num_classes = 1000);
std::unique_ptr<Network> create_simple_cnn(const std::vector<int> &input_shape,
                                           int num_classes);

} // namespace CNN

#endif // CNN_NETWORK_H_