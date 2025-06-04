# C语言核心库
file(GLOB_RECURSE C_CORE_SOURCES "core_c/*.c")

# C++封装层源文件
file(GLOB_RECURSE CPP_SOURCES "cpp/*.cpp")

# 如果有CUDA支持，添加CUDA源文件
if(CUDA_FOUND)
    file(GLOB_RECURSE CUDA_SOURCES "core_c/*.cu")
    list(APPEND C_CORE_SOURCES ${CUDA_SOURCES})
endif()

# 创建C核心库
add_library(cnn_core_c STATIC ${C_CORE_SOURCES})

# 设置C库属性
set_target_properties(cnn_core_c PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# C核心库的编译选项
target_compile_options(cnn_core_c PRIVATE -O3)
if(MINGW)
    target_compile_options(cnn_core_c PRIVATE -march=native -fopenmp)
endif()

# C核心库的包含目录
target_include_directories(cnn_core_c PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# C核心库链接OpenMP
if(OpenMP_C_FOUND)
    target_link_libraries(cnn_core_c PUBLIC OpenMP::OpenMP_C)
endif()

# 如果找到OpenBLAS，链接它
if(OPENBLAS_FOUND)
    target_link_libraries(cnn_core_c PUBLIC ${OPENBLAS_LIBRARIES})
    target_include_directories(cnn_core_c PUBLIC ${OPENBLAS_INCLUDE_DIRS})
    target_compile_definitions(cnn_core_c PUBLIC USE_OPENBLAS)
endif()

# 如果有CUDA支持
if(CUDA_FOUND)
    target_link_libraries(cnn_core_c PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
    target_include_directories(cnn_core_c PUBLIC ${CUDA_INCLUDE_DIRS})
    target_compile_definitions(cnn_core_c PUBLIC USE_CUDA)
    
    # 设置CUDA属性
    set_target_properties(cnn_core_c PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

# 创建C++封装库
add_library(cnn_core_cpp STATIC ${CPP_SOURCES})

# 设置C++库属性
set_target_properties(cnn_core_cpp PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# C++封装库的包含目录
target_include_directories(cnn_core_cpp PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# C++封装库链接C核心库
target_link_libraries(cnn_core_cpp PUBLIC cnn_core_c)

# C++封装库链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(cnn_core_cpp PUBLIC OpenMP::OpenMP_CXX)
endif()

# 创建最终的cnn_core库（向后兼容）
add_library(cnn_core INTERFACE)
target_link_libraries(cnn_core INTERFACE cnn_core_cpp)

# 编译器特定设置
if(MSVC)
    target_compile_options(cnn_core_c PRIVATE /W3)
    target_compile_options(cnn_core_cpp PRIVATE /W4)
else()
    target_compile_options(cnn_core_c PRIVATE -Wall -Wextra)
    target_compile_options(cnn_core_cpp PRIVATE -Wall -Wextra)
endif()

# 安装库
install(TARGETS cnn_core_c cnn_core_cpp cnn_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
) 