# 测试目录的CMakeLists.txt

# Google Test配置 - 仅支持vcpkg安装的版本
find_package(GTest CONFIG QUIET)

if(GTest_FOUND)
    message(STATUS "使用vcpkg安装的Google Test")
    
    # 对于MinGW，需要特别的链接配置
    if(MINGW)
        set(GTEST_LIBRARIES GTest::gtest_main GTest::gtest)
        # MinGW需要额外的系统库
        if(WIN32)
            set(GTEST_SYSTEM_LIBS ws2_32 wsock32)
        endif()
    else()
        set(GTEST_LIBRARIES GTest::gtest GTest::gtest_main)
        set(GTEST_SYSTEM_LIBS "")
    endif()
    
    set(GTEST_INCLUDE_DIRS "")
    
    # 包含GoogleTest模块
    include(GoogleTest)
    
    # 检查测试源文件是否存在并创建测试
    set(TEST_TENSOR_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_tensor.cpp")
    set(TEST_NETWORK_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_network.cpp")
    set(TEST_LAYERS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test_layers.cpp")
    
    # 张量测试
    if(EXISTS "${TEST_TENSOR_SRC}")
        add_executable(test_tensor ${TEST_TENSOR_SRC})
        target_link_libraries(test_tensor 
            PRIVATE 
            cnn 
            ${GTEST_LIBRARIES}
            ${GTEST_SYSTEM_LIBS}
        )
        target_include_directories(test_tensor PRIVATE 
            ${CMAKE_SOURCE_DIR}/include
            ${GTEST_INCLUDE_DIRS}
        )
        gtest_discover_tests(test_tensor)
        message(STATUS "添加测试: test_tensor")
    endif()
    
    # 网络测试
    if(EXISTS "${TEST_NETWORK_SRC}")
        add_executable(test_network ${TEST_NETWORK_SRC})
        target_link_libraries(test_network 
            PRIVATE 
            cnn 
            ${GTEST_LIBRARIES}
            ${GTEST_SYSTEM_LIBS}
        )
        target_include_directories(test_network PRIVATE 
            ${CMAKE_SOURCE_DIR}/include
            ${GTEST_INCLUDE_DIRS}
        )
        gtest_discover_tests(test_network)
        message(STATUS "添加测试: test_network")
    endif()
    
    # 层测试
    if(EXISTS "${TEST_LAYERS_SRC}")
        add_executable(test_layers ${TEST_LAYERS_SRC})
        target_link_libraries(test_layers 
            PRIVATE 
            cnn 
            ${GTEST_LIBRARIES}
            ${GTEST_SYSTEM_LIBS}
        )
        target_include_directories(test_layers PRIVATE 
            ${CMAKE_SOURCE_DIR}/include
            ${GTEST_INCLUDE_DIRS}
        )
        gtest_discover_tests(test_layers)
        message(STATUS "添加测试: test_layers")
    endif()
    
    # 安装配置
    set(TEST_TARGETS_TO_INSTALL "")
    if(EXISTS "${TEST_TENSOR_SRC}")
        list(APPEND TEST_TARGETS_TO_INSTALL test_tensor)
    endif()
    if(EXISTS "${TEST_NETWORK_SRC}")
        list(APPEND TEST_TARGETS_TO_INSTALL test_network)
    endif()
    if(EXISTS "${TEST_LAYERS_SRC}")
        list(APPEND TEST_TARGETS_TO_INSTALL test_layers)
    endif()
    
    if(TEST_TARGETS_TO_INSTALL)
        install(TARGETS ${TEST_TARGETS_TO_INSTALL}
            RUNTIME DESTINATION bin/tests
        )
    endif()
    
else()
    message(WARNING "Google Test未找到，跳过测试构建")
    message(STATUS "如需构建测试，请通过vcpkg安装: vcpkg install gtest:x64-mingw-static")
endif() 