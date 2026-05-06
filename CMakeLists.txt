cmake_minimum_required(VERSION 3.18)
project(cg_poisson LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)

option(ENABLE_MPI "Enable MPI support" ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_CUDA "Enable CUDA support" ON)

if(ENABLE_MPI)
  find_package(MPI QUIET)
  if(NOT MPI_C_FOUND)
    message(WARNING "MPI not found; building without MPI support. Set ENABLE_MPI=OFF to silence.")
    set(ENABLE_MPI OFF)
  endif()
endif()

if(ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

if(ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

set(CG_SOLVER_SOURCES
  src/main.c
  src/poisson_setup.c
  src/cg_serial.c
  src/cg_omp.c
  src/cg_cuda.cu
  src/perf_utils.c
  cuda/spmv_csr.cu
  cuda/cuda_utils.cu
)

if(ENABLE_MPI)
  list(APPEND CG_SOLVER_SOURCES src/cg_mpi.c src/cg_mpi_cuda.cu)
endif()

add_executable(cg_solver ${CG_SOLVER_SOURCES})

target_compile_options(cg_solver PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
  $<$<COMPILE_LANGUAGE:C>:-O3 -march=native>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

target_link_libraries(cg_solver PRIVATE m)

if(ENABLE_MPI)
  target_link_libraries(cg_solver PRIVATE MPI::MPI_C)
  target_compile_definitions(cg_solver PRIVATE ENABLE_MPI)
endif()
if(ENABLE_OPENMP)
  target_link_libraries(cg_solver PRIVATE OpenMP::OpenMP_C)
endif()
if(ENABLE_CUDA)
  target_link_libraries(cg_solver PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
  target_compile_definitions(cg_solver PRIVATE ENABLE_CUDA)
endif()

add_executable(test_cg tests/test_cg.c src/poisson_setup.c src/cg_serial.c src/cg_omp.c src/perf_utils.c)
target_link_libraries(test_cg PRIVATE m)
if(ENABLE_OPENMP)
  target_link_libraries(test_cg PRIVATE OpenMP::OpenMP_C)
endif()

add_executable(test_spmv tests/test_spmv.cu cuda/spmv_csr.cu cuda/cuda_utils.cu src/poisson_setup.c)
target_link_libraries(test_spmv PRIVATE m)
if(ENABLE_CUDA)
  target_link_libraries(test_spmv PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse)
endif()
