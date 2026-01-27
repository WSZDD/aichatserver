cmake_minimum_required(VERSION 3.5.0)
project(LlamaWrapper)

# ==============================================================================
# 1. 基础设置
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(LLAMA_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp)
set(SHERPA_NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/sherpa-ncnn)

# ==============================================================================
# 2. Sherpa-ncnn 配置
# ==============================================================================
include(FetchContent)
FetchContent_Declare(kissfft SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/kissfft")
FetchContent_Declare(kaldi_native_fbank SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/kaldi-native-fbank")
FetchContent_Declare(kaldifst SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/kaldifst")
FetchContent_Declare(openfst SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/openfst")
FetchContent_Declare(ncnn SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/ncnn")

set(SHERPA_NCNN_ENABLE_PORTAUDIO OFF CACHE BOOL "" FORCE)
set(SHERPA_NCNN_ENABLE_JNI OFF CACHE BOOL "" FORCE)
set(SHERPA_NCNN_ENABLE_C_API ON CACHE BOOL "" FORCE)
set(SHERPA_NCNN_ENABLE_BINARY OFF CACHE BOOL "" FORCE)
set(SHERPA_NCNN_ENABLE_WEBSOCKET OFF CACHE BOOL "" FORCE)

list(APPEND CMAKE_MODULE_PATH "${SHERPA_NCNN_DIR}/cmake")
add_subdirectory(${SHERPA_NCNN_DIR})

include_directories(${SHERPA_NCNN_DIR})
include_directories(${SHERPA_NCNN_DIR}/sherpa-ncnn/c-api)
target_include_directories(sherpa-ncnn-core PUBLIC ${SHERPA_NCNN_DIR})
target_include_directories(sherpa-ncnn-c-api PUBLIC ${SHERPA_NCNN_DIR})

# ==============================================================================
# 3. LLM 核心配置 (您的代码)
# ==============================================================================
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
add_definitions(-DNDEBUG)

# 优化选项
add_compile_options(-O3)
add_compile_options(-funroll-loops)

# 🔥🔥🔥 核心修复区：32位架构配置 🔥🔥🔥
if(${OHOS_ARCH} STREQUAL "armeabi-v7a")
    # 您的原版配置
    add_definitions(-mfpu=neon-fp16)
    add_compile_options(-mfloat-abi=softfp)
    add_compile_options(-march=armv7-a)

    # 🟢 新增：尝试骗过 ggml-cpu-impl.h，让它不要重定义
    # 这两个宏通常用来指示系统已经提供了 FP16 支持
    add_definitions(-D__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    add_definitions(-D__ARM_NEON_FP)
endif()

# ==============================================================================
# 4. 生成库
# ==============================================================================
add_library(mnnllm SHARED
    napi_init.cpp
    sherpa_napi.cpp
    ${ALL_SRCS}
)

target_link_libraries(mnnllm PUBLIC
    libace_napi.z.so
    libhilog_ndk.z.so
    sherpa-ncnn-c-api
)