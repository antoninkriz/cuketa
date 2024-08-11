cmake_minimum_required(VERSION 3.28)
project(GPUsemestral CUDA)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -use_fast_math -Xptxas -O3 -Xcompiler -O3 -arch=sm_86 --forward-unknown-to-host-compiler -O3 -mtune=native -march=native")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -G -line-info -g -Xptxas -O0 -Xcompiler -O0 -arch=sm_86 --forward-unknown-to-host-compiler -O0 -g -mtune=native -march=native")

add_executable(GPUsemestral main.cu utils.cuh)

set_target_properties(GPUsemestral PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
