# C++示例的CMakeLists.txt

# 添加MNIST训练示例
add_executable(mnist_training mnist_training.cpp)
target_link_libraries(mnist_training PRIVATE cnn)

# 创建更多示例
# 简单示例
add_executable(simple_cnn simple_cnn.cpp)
target_link_libraries(simple_cnn PRIVATE cnn)

# 安装配置
install(TARGETS 
    mnist_training
    simple_cnn
    RUNTIME DESTINATION bin/examples
) 