# Python绑定的CMakeLists.txt

# 设置源文件
set(PYTHON_SOURCES
    cnn_module.cpp
    tensor_binding.cpp
    network_binding.cpp
    layers_binding.cpp
)

# 创建Python模块
pybind11_add_module(cnn_framework ${PYTHON_SOURCES})

# 设置包含路径
target_include_directories(cnn_framework PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${pybind11_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
)

# 设置依赖关系
target_link_libraries(cnn_framework PRIVATE 
    cnn
    cnn_core
)

# 添加编译定义
target_compile_definitions(cnn_framework PRIVATE
    CNN_PYTHON_BUILDING
)

# 如果启用了OpenMP
if(OpenMP_FOUND)
    target_link_libraries(cnn_framework PRIVATE OpenMP::OpenMP_CXX)
endif()

# 如果启用了OpenBLAS
if(BLAS_FOUND)
    target_link_libraries(cnn_framework PRIVATE ${BLAS_LIBRARIES})
endif()

# 如果启用了CUDA
if(CUDA_FOUND)
    target_link_libraries(cnn_framework PRIVATE ${CUDA_LIBRARIES})
    target_include_directories(cnn_framework PRIVATE ${CUDA_INCLUDE_DIRS})
endif()

# 设置输出目录
set_target_properties(cnn_framework PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_PYTHON_OUTPUT_DIRECTORY}
)

# 安装配置
install(TARGETS cnn_framework
    LIBRARY DESTINATION lib/python
) 