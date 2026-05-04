cmake_minimum_required(VERSION 3.18)

# CMP0141: suppress MSVC debug-info abstraction so -Fd/-FS are not injected.
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "")
endif()

# ---------------------------------------------------------------------------
# Project declaration
# ---------------------------------------------------------------------------
project(DecisionTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MINGW)
    add_compile_options(-pipe)
endif()

# ---------------------------------------------------------------------------
# OpenMP
# ---------------------------------------------------------------------------
option(ENABLE_OPENMP "Enable OpenMP parallelism for level-wise tree building" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found — parallel level-wise expansion enabled")
else()
    message(STATUS "OpenMP disabled — sequential build")
endif()

# ---------------------------------------------------------------------------
# CUDA
# Opt-in: cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ..
#   89 = RTX 40xx (Ada Lovelace)  86 = RTX 30xx  75 = RTX 20xx
# ---------------------------------------------------------------------------
option(ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

if(ENABLE_CUDA)
    # CMake 4.x + MSVC + Ninja injects -Fd/-FS into nvcc via comma-separated
    # -Xcompiler which nvcc misparses as multiple input files.  Work around by
    # compiling split_kernel.cu with an explicit add_custom_command that calls
    # nvcc directly with only the flags we control.
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA target GPU architecture")
    add_compile_definitions(USE_CUDA)
    message(STATUS "CUDA enabled — architecture: ${CMAKE_CUDA_ARCHITECTURES}")

    set(CUDA_KERNEL_SRC "${CMAKE_SOURCE_DIR}/src/gpu/split_kernel.cu")
    set(CUDA_KERNEL_OBJ "${CMAKE_BINARY_DIR}/split_kernel_cuda.obj")

    # Explicit nvcc invocation — no CMake-injected MSVC flags.
    add_custom_command(
        OUTPUT  "${CUDA_KERNEL_OBJ}"
        COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                -x cu -c "${CUDA_KERNEL_SRC}"
                -o "${CUDA_KERNEL_OBJ}"
                -I "${CMAKE_SOURCE_DIR}/src"
                -I "${CMAKE_SOURCE_DIR}/eval"
                -DUSE_CUDA -DUSE_OPENMP
                -std=c++17
                -O2
                -Xcompiler=/MD
                --generate-code arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=[compute_${CMAKE_CUDA_ARCHITECTURES},sm_${CMAKE_CUDA_ARCHITECTURES}]
        DEPENDS "${CUDA_KERNEL_SRC}"
                "${CMAKE_SOURCE_DIR}/src/gpu/split_kernel.cuh"
        COMMENT "Compiling CUDA kernel: split_kernel.cu"
        VERBATIM
    )
else()
    message(STATUS "CUDA disabled — CPU-only build")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/tree/decision_tree.cpp
)

if(ENABLE_CUDA)
    list(APPEND SOURCES "${CUDA_KERNEL_OBJ}")
endif()

add_executable(decision_tree ${SOURCES})

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(decision_tree PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/eval
)

# ---------------------------------------------------------------------------
# Link OpenMP
# ---------------------------------------------------------------------------
if(ENABLE_OPENMP)
    target_compile_definitions(decision_tree PRIVATE USE_OPENMP)
    target_link_libraries(decision_tree PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Link CUDA runtime
# ---------------------------------------------------------------------------
if(ENABLE_CUDA)
    target_link_libraries(decision_tree PRIVATE CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# Compiler warnings (CXX only — CUDA file compiled via custom command above)
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(decision_tree PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(decision_tree PRIVATE -Wall -Wextra -Wpedantic)
endif()
