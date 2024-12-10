#ifndef CONVOLVE_SIGMA_ADD_F_HPP_
#define CONVOLVE_SIGMA_ADD_F_HPP_

#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>
#if CUDA_AVAILABLE
#include <cuda/convolve_sigma_add_f_cuda.hpp>
#endif

#ifndef COUPLED_SLICE
#define COUPLED_SLICE(n_coupled, start_value, memory_location)                 \
  zisa::array_const_view<Scalar, 1> {                                          \
    zisa::shape_t<1>(n_coupled), &start_value, memory_location                 \
  }
#endif

template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f_cpu(zisa::array_view<Scalar, 2> dst,
                              zisa::array_const_view<Scalar, 2> src,
                              zisa::array_const_view<Scalar, 2> sigma,
                              Scalar del_x_2 /* 1/(dx^2)*/,
                              Scalar del_y_2 /*1/(dx^2)*/, const Function &f) {
  const unsigned Nx = src.shape(0);
  const unsigned Ny = src.shape(1);
  for (int x = 1; x < Nx - 1; x++) {
    for (int y = n_coupled; y < Ny - n_coupled; y += n_coupled) {
      // calculates all f_1, f_2, ... , fn
      // in one run
      Scalar result_function[n_coupled];
      f(COUPLED_SLICE(n_coupled, src(x, y), zisa::device_type::cpu),
        result_function);
#pragma unroll
      for (int i = 0; i < n_coupled; i++) {
        dst(x, y + i) =
            del_x_2 * (sigma(2 * x - 2, y / n_coupled) *
                           (src(x - 1, y + i) - src(x, y + i)) +
                       sigma(2 * x, y / n_coupled) *
                           (src(x + 1, y + i) - src(x, y + i))) +
            del_y_2 * (sigma(2 * x - 1, y / n_coupled - 1) *
                           (src(x, y + i - n_coupled) - src(x, y + i)) +
                       sigma(2 * x - 1, y / n_coupled + 1) *
                           (src(x, y + i + n_coupled) - src(x, y + i))) +
            result_function[i];
      }
    }
  }
}

// apply single function such as
// sin(x), cos(x), x + x³, ...
template <int n_coupled, typename Scalar, typename Function>
void apply_function_cpu(zisa::array_view<Scalar, 2> dst,
                        zisa::array_const_view<Scalar, 2> src,
                        const Function &f) {
  const auto [Nx, Ny] = src.shape;
  for (uint32_t x = 0; x < Nx; ++x) {
    for (uint32_t y = n_coupled; y < Ny - n_coupled; y += n_coupled) {
      Scalar result_function[n_coupled];
      f(COUPLED_SLICE(n_coupled, src(x, y), zisa::device_type::cpu),
        result_function);
#pragma unroll
      for (uint32_t i = 0; i < n_coupled; i++) {
        dst(x, y + i) = result_function[i];
      }
    }
  }
}

// Function is a general function taking a Scalar returning a Scalar
template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f(zisa::array_view<Scalar, 2> dst,
                          zisa::array_const_view<Scalar, 2> src,
                          zisa::array_const_view<Scalar, 2> sigma,
                          Scalar del_x_2 /* 1/(dx^2)*/,
                          Scalar del_y_2 /*1/(dx^2)*/, const Function &f) {

  const zisa::device_type memory_dst = dst.memory_location();
  const zisa::device_type memory_src = src.memory_location();
  const zisa::device_type memory_sigma = sigma.memory_location();
  // if (memory_dst == zisa::device_type::cpu) {
  //   std::cout << "convolve_sigma_add_f called with memory_location = cpu" <<
  //   std::endl;
  // } else if (memory_dst == zisa::device_type::cuda) {
  //   std::cout << "convolve_sigma_add_f called with memory_location = cuda" <<
  //   std::endl;
  // }

  if (!(memory_dst == memory_src && memory_src == memory_sigma)) {
    std::cerr << "Convolve sigma add f: Inputs must be located on the same "
                 "hardware\n";
    exit(1);
  }
  if (dst.shape() != src.shape()) {
    std::cerr << "Convolve sigma add f: Input and output array must have the "
                 "same shape\n";
    exit(1);
  }

  if (memory_dst == zisa::device_type::cpu) {
    convolve_sigma_add_f_cpu<n_coupled>(dst, src, sigma, del_x_2, del_y_2, f);
  }
#if CUDA_AVAILABLE
  else if (memory_dst == zisa::device_type::cuda) {
    convolve_sigma_add_f_cuda<n_coupled>(dst, src, sigma, del_x_2, del_y_2, f);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "Convolution: Unknown device_type of inputs\n";
    exit(1);
  }
}

#endif // CONVOLVE_SIGMA_ADD_F_HPP_