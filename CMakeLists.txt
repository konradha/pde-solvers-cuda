cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(PDE_SOLVERS_CUDA VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g")
option(ENABLE_CUDA "Enable CUDA Acceleration" OFF)

include(CTest)

set(CUDA_AVAILABLE ${ENABLE_CUDA})
set(ZISA_HAS_CUDA ${ENABLE_CUDA})
set(ZISA_HAS_NETCDF ON)
set(ZISA_HAS_HDF5 ON)

# Print current settings for debugging
message(STATUS "Enable CUDA: ${ENABLE_CUDA}")
message(STATUS "C++ Compiler Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA Compiler Flags: ${CMAKE_CUDA_FLAGS}")

if(CUDA_AVAILABLE)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.7
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaCore
  GIT_REPOSITORY https://github.com/1uc/ZisaCore.git
  GIT_TAG 9243c391b70f5b1ff8c0275905351158c4f59eab
  CMAKE_ARGS -DZISA_HAS_CUDA=${ENABLE_CUDA} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  ZisaMemory
  GIT_REPOSITORY https://github.com/1uc/ZisaMemory.git
  GIT_TAG 321013db1830fb77589e13139400aeaea8796f79
  CMAKE_ARGS -DZISA_HAS_CUDA=${ENABLE_CUDA} -DZISA_HAS_NETCDF=1 -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
  OVERRIDE_FIND_PACKAGE
)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.13.0
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(Catch2 ZisaCore ZisaMemory googletest)

find_package(ZisaCore REQUIRED)
find_package(ZisaMemory REQUIRED)
find_package(googletest REQUIRED)

######### netcdf executable start #############
add_executable(run_from_netcdf)
set_target_properties(run_from_netcdf PROPERTIES CXX_STANDARD 17)

if (CUDA_AVAILABLE)
  target_compile_options(run_from_netcdf PUBLIC -DCUDA_AVAILABLE=1)
  target_compile_options(run_from_netcdf PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>>)
endif()

target_include_directories(run_from_netcdf PUBLIC include)

target_link_libraries(run_from_netcdf Zisa::core Zisa::memory)
######### netcdf executable end ###############
######### test executable start ###############
add_executable(unit_tests)
set_target_properties(unit_tests PROPERTIES CXX_STANDARD 17)

if (CUDA_AVAILABLE)
  target_compile_options(unit_tests PUBLIC -DCUDA_AVAILABLE=1)
  target_compile_options(unit_tests PUBLIC $<BUILD_INTERFACE:$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>>)
endif()

target_include_directories(unit_tests PUBLIC include include/kernel include/cuda)

target_link_libraries(unit_tests Zisa::core Zisa::memory gtest_main)
######### test executable end #################

add_subdirectory(src)
add_subdirectory(test)
