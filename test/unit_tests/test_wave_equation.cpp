#include "zisa/memory/device_type.hpp"
#include <generic_function.hpp>
#include <gtest/gtest.h>
#include <pde_wave.hpp>
#include <zisa/memory/array.hpp>

// TODO: add tests for neumann and periodic bc, larger and nonsymmetric grids
//       add tests for sigma != constant (how to get solution)

namespace WaveEquationTests {

// helper function which creates simple data array where all values are set to value,
// if CUDA_AVAILABLE on gpu, else on cpu
template <typename Scalar>
inline zisa::array<Scalar, 2>
create_value_data(int x_size, int y_size, Scalar value,
                  zisa::device_type memory_location) {
  zisa::array<Scalar, 2> data(zisa::shape_t<2>(x_size, y_size),
                              zisa::device_type::cpu);
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      data(i, j) = value;
    }
  }
  if (memory_location == zisa::device_type::cpu) {
    return data;
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    zisa::array<Scalar, 2> data_gpu(zisa::shape_t<2>(x_size, y_size),
                                    zisa::device_type::cuda);
    zisa::copy(data_gpu, data);
    return data_gpu;
  }
#endif
  else {
    std::cout << "device type not supported yet in test_wave_equation"
              << std::endl;
    exit(-1);
  }
}
// helper function which creates simple data array where all arr(i, j) = i*j
template <typename Scalar>
inline zisa::array<Scalar, 2>
create_simple_data(int x_size, int y_size,
                  zisa::device_type memory_location) {
  zisa::array<Scalar, 2> data(zisa::shape_t<2>(x_size, y_size),
                              zisa::device_type::cpu);
  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      data(i, j) = i * j;
    }
  }
  if (memory_location == zisa::device_type::cpu) {
    return data;
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    zisa::array<Scalar, 2> data_gpu(zisa::shape_t<2>(x_size, y_size),
                                    zisa::device_type::cuda);
    zisa::copy(data_gpu, data);
    return data_gpu;
  }
#endif
  else {
    std::cout << "device type not supported yet in test_wave_equation"
              << std::endl;
    exit(-1);
  }
}


// u(x, y, 0) = 0, f = 0, df = 0, sigma = 0 => u(x, y, t) = 0
TEST(WaveEquationTests, TEST_ZERO) {
  const int array_size = 10; // 2 border values included
  #if CUDA_AVAILABLE
  const zisa::device_type memory_location = zisa::device_type::cuda;
  #else
  const zisa::device_type memory_location = zisa::device_type::cpu;
  #endif
  zisa::array<float, 2> data =
      create_value_data<float>(array_size, array_size, 0., memory_location);

  zisa::array<float, 2> sigma_values = create_value_data<float>(
      2 * array_size - 3, array_size - 1, 1., memory_location);
  // f == 0 everywhere
  GenericFunction<float> func;

  PDEWave<float, GenericFunction<float>> pde(
      8, 8, memory_location, BoundaryCondition::Dirichlet, func, 0.1, 0.1);
  pde.read_values(data.const_view(), sigma_values.const_view(),
                  data.const_view());
  for (int i = 0; i < 1000; i++) {
    pde.apply(0.1);
  }
  #if CUDA_AVAILABLE
  zisa::array_const_view<float, 2> result_gpu = pde.get_data();
  zisa::array<float, 2> result(result_gpu.shape());
  zisa::copy(result, result_gpu);
  #else
  zisa::array_const_view<float, 2> result = pde.get_data();
  #endif
  float tol = 1e-10;
  for (int i = 0; i < 10; i++) {
    for (int j = 0; j < 10; j++) {
      ASSERT_NEAR(0.0, result(i, j), tol);
    }
  }
}
} // namespace WaveEquationTests