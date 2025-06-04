# C++封装库的CMakeLists.txt

# 设置源文件
set(CPP_SOURCES
    tensor.cpp
    network.cpp
    layers.cpp
    optimizer.cpp
    loss.cpp
    utils.cpp
)

# 创建C++封装静态库
add_library(cnn STATIC ${CPP_SOURCES})

# 设置包含路径
target_include_directories(cnn PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 设置依赖关系
target_link_libraries(cnn PUBLIC cnn_core)

# 添加编译定义
target_compile_definitions(cnn PRIVATE
    CNN_BUILDING
)

# 如果启用了OpenMP
if(OpenMP_FOUND)
    target_link_libraries(cnn PUBLIC OpenMP::OpenMP_CXX)
endif()

# 如果启用了OpenBLAS
if(BLAS_FOUND)
    target_link_libraries(cnn PUBLIC ${BLAS_LIBRARIES})
endif()

# 如果启用了CUDA
if(CUDA_FOUND)
    target_link_libraries(cnn PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(cnn PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

# 安装配置
install(TARGETS cnn
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 