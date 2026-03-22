cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Project declaration
# LANGUAGES: CXX only for Milestone 1.
# CUDA is scaffolded here and activated via -DENABLE_CUDA=ON in Milestone 2.
# ---------------------------------------------------------------------------
project(DecisionTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MinGW on Windows sometimes fails to create temp files in C:\Windows\.
# -pipe avoids intermediate temp files by using pipes between compiler passes.
if(MINGW)
    add_compile_options(-pipe)
endif()

# ---------------------------------------------------------------------------
# Optional CUDA support (Milestone 2+)
# Teammates: flip this ON when adding GPU kernels in src/gpu/split_kernel.cu.
# Usage: cmake -DENABLE_CUDA=ON ..
# ---------------------------------------------------------------------------
option(ENABLE_CUDA "Enable CUDA GPU acceleration (Milestone 2+)" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    # Adjust CMAKE_CUDA_ARCHITECTURES to match your GPU:
    #   70 = V100, 75 = RTX 20xx, 86 = RTX 30xx, 89 = RTX 40xx
    set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA target architecture")
    add_compile_definitions(USE_CUDA)
    message(STATUS "CUDA enabled — architecture: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled — building CPU-only (Milestone 1)")
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/tree/decision_tree.cpp
)

# Include the CUDA stub only when CUDA is active.
# Milestone 1: the stub is excluded so no CUDA toolchain is required.
if(ENABLE_CUDA)
    list(APPEND SOURCES src/gpu/split_kernel.cu)
endif()

add_executable(decision_tree ${SOURCES})

# ---------------------------------------------------------------------------
# Include directories
# src/   — picks up data_loader.h, and tree/* via relative paths
# eval/  — picks up metrics.h
# ---------------------------------------------------------------------------
target_include_directories(decision_tree PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/eval
)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(decision_tree PRIVATE /W4)
else()
    target_compile_options(decision_tree PRIVATE -Wall -Wextra -Wpedantic)
endif()
