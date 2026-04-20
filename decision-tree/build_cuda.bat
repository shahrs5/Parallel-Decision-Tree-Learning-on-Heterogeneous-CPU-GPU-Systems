@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" > build_cuda_log.txt 2>&1
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin
echo --- CMake Configure --- >> build_cuda_log.txt
cmake -S . -B build_cuda -G "NMake Makefiles" -DENABLE_OPENMP=ON -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 >> build_cuda_log.txt 2>&1
echo --- CMake Build --- >> build_cuda_log.txt
cmake --build build_cuda --config Release >> build_cuda_log.txt 2>&1
echo --- Done --- >> build_cuda_log.txt
