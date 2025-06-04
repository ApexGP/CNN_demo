# MinGW 工具链文件
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# 设置编译器
set(CMAKE_C_COMPILER "C:/Users/Domin/scoop/apps/mingw/current/bin/gcc.exe")
set(CMAKE_CXX_COMPILER "C:/Users/Domin/scoop/apps/mingw/current/bin/g++.exe")

# 设置vcpkg三元组
set(VCPKG_TARGET_TRIPLET "x64-mingw-static")

# 查找vcpkg根目录
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "使用vcpkg工具链: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARNING "未找到VCPKG_ROOT环境变量")
endif()

# MinGW特定的编译标志
set(CMAKE_C_FLAGS_INIT "-Wall")
set(CMAKE_CXX_FLAGS_INIT "-Wall")

# 确保静态链接
set(CMAKE_FIND_LIBRARY_SUFFIXES .a .lib) 