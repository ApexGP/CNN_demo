# C++示例的CMakeLists.txt

# 获取所有源文件
file(GLOB EXAMPLE_SOURCES "*.cpp")

# 为每个源文件创建可执行文件
foreach(SOURCE_FILE ${EXAMPLE_SOURCES})
    # 获取文件名（不含扩展名）
    get_filename_component(EXAMPLE_NAME ${SOURCE_FILE} NAME_WE)
    
    # 创建可执行文件
    add_executable(${EXAMPLE_NAME} ${SOURCE_FILE})
    
    # 链接库
    target_link_libraries(${EXAMPLE_NAME} PRIVATE cnn cnn_core)
    
    # 为需要filesystem的示例添加filesystem库支持
    if("${EXAMPLE_NAME}" STREQUAL "mnist_training")
        # 对于GCC 8及以下版本，需要显式链接stdc++fs
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
            target_link_libraries(${EXAMPLE_NAME} PRIVATE stdc++fs)
        endif()
        
        # 对于Clang某些版本也需要
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
            target_link_libraries(${EXAMPLE_NAME} PRIVATE c++fs)
        endif()
    endif()
    
    # 如果启用了OpenMP
    if(OpenMP_FOUND)
        target_link_libraries(${EXAMPLE_NAME} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    
    # 如果启用了OpenBLAS
    if(BLAS_FOUND)
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ${BLAS_LIBRARIES})
    endif()
    
    # 如果启用了CUDA
    if(CUDA_FOUND)
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ${CUDA_LIBRARIES})
        target_include_directories(${EXAMPLE_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
    endif()
    
    # 设置输出目录
    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    
    message(STATUS "添加C++示例: ${EXAMPLE_NAME}")
endforeach()

# 安装示例
install(TARGETS ${EXAMPLE_TARGETS}
    RUNTIME DESTINATION bin
) 