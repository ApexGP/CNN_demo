# C语言核心库的CMakeLists.txt

# 设置源文件
set(CORE_C_SOURCES
    tensor_core.c
    math_core.c
    conv_core.c
)

# 创建C核心静态库
add_library(cnn_core STATIC ${CORE_C_SOURCES})

# 设置包含路径
target_include_directories(cnn_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# 添加编译定义
target_compile_definitions(cnn_core PRIVATE
    CNN_CORE_BUILDING
)

# 如果启用了OpenMP
if(OpenMP_FOUND)
    target_link_libraries(cnn_core PUBLIC OpenMP::OpenMP_C)
endif()

# 如果启用了OpenBLAS
if(BLAS_FOUND)
    target_link_libraries(cnn_core PUBLIC ${BLAS_LIBRARIES})
endif()

# 如果启用了CUDA
if(CUDA_FOUND)
    target_link_libraries(cnn_core PUBLIC ${CUDA_LIBRARIES})
    target_include_directories(cnn_core PUBLIC ${CUDA_INCLUDE_DIRS})
endif()

# 安装配置
install(TARGETS cnn_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 