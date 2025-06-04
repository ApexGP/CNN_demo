# API 参考文档

## C++ API

### Tensor 类

#### 构造函数

```cpp
// 默认构造函数
Tensor();

// 指定形状和数据类型
Tensor(const std::vector<int>& shape, DataType dtype = DataType::FLOAT32, Device device = Device::CPU);

// 从数据构造
Tensor(const std::vector<int>& shape, const std::vector<float>& data, Device device = Device::CPU);
```

#### 基本属性

```cpp
const std::vector<int>& shape() const;  // 获取形状
int ndim() const;                       // 获取维度数
int size() const;                       // 获取元素总数
Device device() const;                  // 获取设备类型
DataType dtype() const;                 // 获取数据类型
```

#### 数据访问

```cpp
float* data();                                              // 获取数据指针
const float* data() const;                                  // 获取只读数据指针
float& operator()(const std::vector<int>& indices);        // 多维索引访问
const float& operator()(const std::vector<int>& indices) const;
float& at(int i);                                          // 一维索引访问
const float& at(int i) const;
```

#### 数学运算

```cpp
Tensor operator+(const Tensor& other) const;               // 张量加法
Tensor operator-(const Tensor& other) const;               // 张量减法
Tensor operator*(const Tensor& other) const;               // 元素乘法
Tensor operator/(const Tensor& other) const;               // 元素除法

Tensor operator+(float scalar) const;                      // 标量加法
Tensor operator*(float scalar) const;                      // 标量乘法

Tensor matmul(const Tensor& other) const;                  // 矩阵乘法
```

#### 激活函数

```cpp
Tensor relu() const;                                        // ReLU激活
Tensor sigmoid() const;                                     // Sigmoid激活
Tensor tanh() const;                                        // Tanh激活
Tensor softmax(int dim = -1) const;                        // Softmax激活
```

#### 初始化方法

```cpp
void zeros();                                               // 全零初始化
void ones();                                                // 全一初始化
void uniform(float low = 0.0f, float high = 1.0f);        // 均匀分布初始化
void normal(float mean = 0.0f, float std = 1.0f);         // 正态分布初始化
void xavier_uniform();                                      // Xavier均匀初始化
void kaiming_uniform();                                     // Kaiming均匀初始化
```

### Network 类

#### 构造函数

```cpp
Network();                                                  // 默认构造函数
```

#### 网络构建

```cpp
template<typename LayerType, typename... Args>
void add_layer(Args&&... args);                           // 添加任意类型的层

// 便捷方法
void add_conv_layer(int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true);
void add_fc_layer(int out_features, bool bias = true);
void add_relu_layer();
void add_sigmoid_layer();
void add_tanh_layer();
void add_softmax_layer(int dim = -1);
void add_maxpool_layer(int kernel_size, int stride = -1, int padding = 0);
void add_avgpool_layer(int kernel_size, int stride = -1, int padding = 0);
void add_dropout_layer(float p = 0.5);
void add_batchnorm_layer(int num_features, float eps = 1e-5, float momentum = 0.1);
void add_flatten_layer();
```

#### 训练和推理

```cpp
// 前向传播
Tensor forward(const Tensor& input);
Tensor predict(const Tensor& input);

// 训练
void train(const std::vector<Tensor>& train_data,
           const std::vector<Tensor>& train_labels,
           int epochs = 100, int batch_size = 32, float learning_rate = 0.001f);

void train_with_validation(const std::vector<Tensor>& train_data,
                          const std::vector<Tensor>& train_labels,
                          const std::vector<Tensor>& val_data,
                          const std::vector<Tensor>& val_labels,
                          int epochs = 100, int batch_size = 32, float learning_rate = 0.001f);

// 评估
float evaluate(const std::vector<Tensor>& test_data, const std::vector<Tensor>& test_labels);
float calculate_accuracy(const std::vector<Tensor>& data, const std::vector<Tensor>& labels);
```

#### 配置设置

```cpp
void set_optimizer(std::unique_ptr<Optimizer> optimizer);
void set_loss_function(std::unique_ptr<LossFunction> loss_fn);
void set_training_mode(bool training);
void train_mode();
void eval_mode();
```

#### 设备管理

```cpp
void to_cpu();
void to_gpu();
Device get_device() const;
```

#### 模型保存和加载

```cpp
void save_model(const std::string& filename) const;
void load_model(const std::string& filename);
void save_weights(const std::string& filename) const;
void load_weights(const std::string& filename);
```

#### 可视化和监控

```cpp
void visualize_training(const std::string& save_path = "") const;
void visualize_network_architecture(const std::vector<int>& input_shape, const std::string& save_path = "") const;
void print_summary(const std::vector<int>& input_shape) const;
```

### Layer 类层次结构

#### 基类 Layer

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Tensor*> parameters();
    virtual std::vector<Tensor*> gradients();
    virtual void train(bool mode = true);
    virtual std::string name() const = 0;
};
```

#### ConvLayer (卷积层)

```cpp
ConvLayer(int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true);
ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0, bool bias = true);
```

#### FullyConnectedLayer (全连接层)

```cpp
FullyConnectedLayer(int out_features, bool bias = true);
FullyConnectedLayer(int in_features, int out_features, bool bias = true);
```

### Optimizer 类层次结构

#### SGDOptimizer

```cpp
SGDOptimizer(float learning_rate = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
```

#### AdamOptimizer

```cpp
AdamOptimizer(float learning_rate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f,
              float eps = 1e-8f, float weight_decay = 0.0f);
```

### LossFunction 类层次结构

#### CrossEntropyLoss

```cpp
CrossEntropyLoss(bool from_logits = true, float label_smoothing = 0.0f);
```

#### MSELoss

```cpp
MSELoss();
```

## Python API

### Tensor 类

```python
import cnn_demo

# 创建张量
tensor = cnn_demo.Tensor([2, 3, 4])

# 基本属性
shape = tensor.shape()
size = tensor.size()

# 打印信息
tensor.print()
```

### Network 类

```python
# 创建网络
net = cnn_demo.Network()

# 添加层
net.add_conv_layer(32, 3, stride=1, padding=1)
net.add_relu_layer()
net.add_maxpool_layer(2)
net.add_flatten_layer()
net.add_fc_layer(10)
net.add_softmax_layer()

# 训练
train_data = [[...], [...], ...]  # 训练数据
train_labels = [0, 1, 2, ...]     # 训练标签
net.train(train_data, train_labels, epochs=50, batch_size=32, learning_rate=0.001)

# 预测
prediction = net.predict([...])

# 可视化
net.visualize_training()

# 打印摘要
net.print_summary()
```

## 工厂函数

### Tensor 工厂函数

```cpp
Tensor zeros(const std::vector<int>& shape, Device device = Device::CPU);
Tensor ones(const std::vector<int>& shape, Device device = Device::CPU);
Tensor randn(const std::vector<int>& shape, Device device = Device::CPU);
Tensor eye(int n, Device device = Device::CPU);
```

### Optimizer 工厂函数

```cpp
std::unique_ptr<Optimizer> create_sgd_optimizer(float lr = 0.01f, float momentum = 0.0f, float weight_decay = 0.0f);
std::unique_ptr<Optimizer> create_adam_optimizer(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f);
```

### LossFunction 工厂函数

```cpp
std::unique_ptr<LossFunction> create_cross_entropy_loss(bool from_logits = true, float label_smoothing = 0.0f);
std::unique_ptr<LossFunction> create_mse_loss();
```

## 预定义网络模板

```cpp
// 创建经典网络架构
std::unique_ptr<Network> create_lenet5(int num_classes = 10);
std::unique_ptr<Network> create_alexnet(int num_classes = 1000);
std::unique_ptr<Network> create_simple_cnn(const std::vector<int>& input_shape, int num_classes);
```

## 错误处理

所有 API 函数可能抛出以下异常：

- `std::invalid_argument`: 无效参数
- `std::out_of_range`: 索引超出范围
- `std::runtime_error`: 运行时错误
- `std::bad_alloc`: 内存分配失败

## 线程安全性

- **线程安全的类**: Tensor (只读操作)
- **非线程安全的类**: Network, Layer (训练过程中)
- **推荐**: 每个线程使用独立的 Network 实例
