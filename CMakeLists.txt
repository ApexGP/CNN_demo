cmake_minimum_required(VERSION 3.15)

# vcpkg工具链支持
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
    message(STATUS "使用vcpkg工具链: ${CMAKE_TOOLCHAIN_FILE}")
endif()

# 如果通过命令行指定了vcpkg路径
if(DEFINED VCPKG_ROOT AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "")
    message(STATUS "使用指定的vcpkg工具链: ${CMAKE_TOOLCHAIN_FILE}")
endif()

project(CNN_DEMO VERSION 0.1.0 LANGUAGES C CXX)

# 生成compile_commands.json供clangd使用
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置C标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 编译选项
option(USE_OPENMP "使用OpenMP加速" ON)
option(USE_OPENBLAS "使用OpenBLAS加速" ON)
option(USE_CUDA "使用CUDA加速" OFF)
option(BUILD_PYTHON "构建Python绑定" ON)
option(BUILD_EXAMPLES "构建示例" ON)
option(BUILD_TESTS "构建测试" ON)

# 设置输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_PYTHON_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# OpenMP支持
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message(STATUS "OpenMP found and enabled")
    else()
        message(WARNING "OpenMP requested but not found")
    endif()
endif()

# OpenBLAS支持
if(USE_OPENBLAS)
    # 方法1: 尝试通过vcpkg查找OpenBLAS (推荐)
    find_package(OpenBLAS CONFIG QUIET)
    if(OpenBLAS_FOUND)
        message(STATUS "找到vcpkg安装的OpenBLAS: ${OpenBLAS_VERSION}")
        # vcpkg的OpenBLAS通常提供OpenBLAS::OpenBLAS目标
        set(BLAS_FOUND TRUE)
        set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
        add_definitions(-DUSE_OPENBLAS)
    else()
        # 方法2: 尝试直接查找BLAS
        find_package(BLAS QUIET)
        
        # 方法3: 如果没找到，尝试在Anaconda环境中找
        if(NOT BLAS_FOUND)
            # 尝试Anaconda路径
            if(WIN32)
                # Windows上的Anaconda路径检查
                if(DEFINED ENV{CONDA_PREFIX})
                    set(ANACONDA_ROOT $ENV{CONDA_PREFIX})
                    message(STATUS "检查Anaconda路径: ${ANACONDA_ROOT}")
                    
                    # 添加Anaconda库路径
                    set(BLAS_LIBRARIES ${ANACONDA_ROOT}/Library/lib/openblas.lib)
                    if(EXISTS ${BLAS_LIBRARIES})
                        set(BLAS_FOUND TRUE)
                        include_directories(${ANACONDA_ROOT}/Library/include)
                        message(STATUS "在Anaconda中找到OpenBLAS: ${BLAS_LIBRARIES}")
                    endif()
                endif()
            else()
                # Linux/Mac上的Anaconda路径检查
                if(DEFINED ENV{CONDA_PREFIX})
                    set(ANACONDA_ROOT $ENV{CONDA_PREFIX})
                    set(BLAS_LIBRARIES ${ANACONDA_ROOT}/lib/libopenblas.so)
                    if(EXISTS ${BLAS_LIBRARIES})
                        set(BLAS_FOUND TRUE)
                        include_directories(${ANACONDA_ROOT}/include)
                        message(STATUS "在Anaconda中找到OpenBLAS: ${BLAS_LIBRARIES}")
                    endif()
                endif()
            endif()
        endif()
        
        if(BLAS_FOUND)
            include_directories(${BLAS_INCLUDE_DIRS})
            link_libraries(${BLAS_LIBRARIES})
            add_definitions(-DUSE_OPENBLAS)
            message(STATUS "BLAS已找到并启用")
        else()
            message(WARNING "请求OpenBLAS但未找到。某些功能将被禁用。")
            message(STATUS "建议安装: vcpkg install openblas")
            # 设置一个宏表示没有OpenBLAS，以便代码可以适配
            add_definitions(-DNO_OPENBLAS)
        endif()
    endif()
endif()

# CUDA支持
if(USE_CUDA)
    find_package(CUDA)
    if(CUDA_FOUND)
        include_directories(${CUDA_INCLUDE_DIRS})
        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA found and enabled")
    else()
        message(WARNING "CUDA requested but not found")
    endif()
endif()

# Python绑定
if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development)
    if(Python_FOUND)
        message(STATUS "Python found: ${Python_VERSION}")
        
        # 查找pybind11
        find_package(pybind11 CONFIG QUIET)
        if(pybind11_FOUND)
            message(STATUS "pybind11 found")
        else()
            # 如果系统没有找到pybind11，尝试使用Python的pybind11
            execute_process(
                COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(PYBIND11_CMAKE_DIR)
                # 将反斜杠转换为正斜杠，避免转义字符问题
                string(REPLACE "\\" "/" PYBIND11_CMAKE_DIR_FIXED "${PYBIND11_CMAKE_DIR}")
                message(STATUS "Found pybind11 via Python: ${PYBIND11_CMAKE_DIR_FIXED}")
                find_package(pybind11 CONFIG QUIET PATHS "${PYBIND11_CMAKE_DIR_FIXED}" NO_DEFAULT_PATH)
                if(NOT pybind11_FOUND)
                    # 如果还是找不到，尝试通过Python直接导入
                    message(STATUS "Trying alternative pybind11 setup...")
                    execute_process(
                        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
                        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET
                    )
                    if(PYBIND11_INCLUDE_DIR)
                        message(STATUS "Using pybind11 include path: ${PYBIND11_INCLUDE_DIR}")
                        # 手动设置pybind11变量
                        set(pybind11_FOUND TRUE)
                        set(pybind11_INCLUDE_DIRS "${PYBIND11_INCLUDE_DIR}")
                    endif()
                endif()
            endif()
            
            if(NOT pybind11_FOUND)
                message(WARNING "pybind11 not found, Python bindings will be disabled")
                set(BUILD_PYTHON OFF)
            endif()
        endif()
    else()
        message(WARNING "Python not found, Python bindings will be disabled")
        set(BUILD_PYTHON OFF)
    endif()
endif()

# 首先构建C核心库
add_subdirectory(src/core_c)

# 然后构建C++封装库
add_subdirectory(src/cpp)

# Python绑定
if(BUILD_PYTHON)
    add_subdirectory(src/python)
endif()

# 构建示例
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 构建测试
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# 安装配置
install(DIRECTORY include/ DESTINATION include)

# 打印配置信息
message(STATUS "--------------------------------------")
message(STATUS "CNN Demo Configuration:")
message(STATUS "  Version:             ${CMAKE_PROJECT_VERSION}")
message(STATUS "  C Compiler:          ${CMAKE_C_COMPILER}")
message(STATUS "  C++ Compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "  Build Type:          ${CMAKE_BUILD_TYPE}")
message(STATUS "  OpenMP:              ${USE_OPENMP}")
message(STATUS "  OpenBLAS:            ${USE_OPENBLAS}")
message(STATUS "  CUDA:                ${USE_CUDA}")
message(STATUS "  Python Bindings:     ${BUILD_PYTHON}")
message(STATUS "--------------------------------------") 