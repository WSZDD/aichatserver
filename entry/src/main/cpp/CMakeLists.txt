cmake_minimum_required(VERSION 3.5.0)
project(LlamaWrapper)

# 基础设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp)

include_directories(
    ${LLAMA_ROOT}/include
    ${LLAMA_ROOT}/src
    ${LLAMA_ROOT}/common
    ${LLAMA_ROOT}/ggml/include
    ${LLAMA_ROOT}/ggml/src
    ${LLAMA_ROOT}/ggml/src/ggml-cpu
)

file(GLOB_RECURSE ALL_SRCS
    "${LLAMA_ROOT}/*.cpp"
    "${LLAMA_ROOT}/*.c"
)

# 过滤黑名单
list(FILTER ALL_SRCS EXCLUDE REGEX ".*main\\.cpp$")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*/common/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*tests/.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-hexagon.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-zdnn.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-zendnn.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-webgpu.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cann.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-blas.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*spacemit.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*kleidiai.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*amx.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-opencl.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cuda.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-vulkan.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-metal.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-rpc.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-kompute.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-sycl.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/x86.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/riscv.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/powerpc.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/s390.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/loongarch.*")
list(FILTER ALL_SRCS EXCLUDE REGEX ".*ggml-cpu/arch/wasm.*")

# 宏定义
add_definitions("-DGGML_VERSION=((char*)0)")
add_definitions("-DGGML_COMMIT=((char*)0)")
add_definitions(-DGGML_USE_CPU)
add_definitions(-D_FILE_OFFSET_BITS=64)
add_definitions(-DGGML_SCHED_MAX_COPIES=4)
add_definitions(-DNDEBUG) # 关闭断言 (重要)

# 开启 O3 最高级优化
add_compile_options(-O3)

# 循环展开优化
add_compile_options(-funroll-loops)

# 32位架构专属优化 (解决 sgemm.cpp 报错)
if(${OHOS_ARCH} STREQUAL "armeabi-v7a")
    # 必须使用 neon-fp16 才能识别 vld1q_f16 指令
    add_definitions(-mfpu=neon-fp16)

    # 指定浮点 ABI
    add_compile_options(-mfloat-abi=softfp)

    # 显式指定架构 (可选，增加稳定性)
    add_compile_options(-march=armv7-a)
endif()

# 生成库
add_library(mnnllm SHARED napi_init.cpp ${ALL_SRCS})
target_link_libraries(mnnllm PUBLIC libace_napi.z.so libhilog_ndk.z.so)