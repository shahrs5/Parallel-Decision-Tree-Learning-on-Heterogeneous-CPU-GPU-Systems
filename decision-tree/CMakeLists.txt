cmake_minimum_required(VERSION 3.18)

# ---------------------------------------------------------------------------
# Project declaration
# Milestone 1: CXX only.
# Milestone 2 (Person 3): OpenMP enabled by default; CUDA opt-in via -DENABLE_CUDA=ON.
# ---------------------------------------------------------------------------
project(DecisionTree LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MINGW)
    add_compile_options(-pipe)
endif()

# ---------------------------------------------------------------------------
# OpenMP (Milestone 2 — Person 3: level-wise parallel node expansion)
# Enabled by default; disable with -DENABLE_OPENMP=OFF
# ---------------------------------------------------------------------------
option(ENABLE_OPENMP "Enable OpenMP parallelism for level-wise tree building" ON)

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found — parallel level-wise expansion enabled")
else()
    message(STATUS "OpenMP disabled — sequential build")
endif()

# ---------------------------------------------------------------------------
# CUDA (Milestone 2 — Person 2/3: GPU-accelerated split finding)
# Opt-in: cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 ..
#   75 = RTX 20xx (Turing)
#   86 = RTX 30xx (Ampere)
#   89 = RTX 40xx (Ada Lovelace) — RTX 4060
# ---------------------------------------------------------------------------
option(ENABLE_CUDA "Enable CUDA GPU acceleration (Milestone 2+)" OFF)

if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA target GPU architecture")
    add_compile_definitions(USE_CUDA)
    message(STATUS "CUDA enabled — architecture: ${CMAKE_CUDA_ARCHITECTURES}")
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
    list(APPEND SOURCES src/gpu/split_kernel.cu)
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
# Compiler warnings
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(decision_tree PRIVATE /W4)
else()
    target_compile_options(decision_tree PRIVATE -Wall -Wextra -Wpedantic)
endif()
