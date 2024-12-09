#ifndef HELPERS_HPP_
#define HELPERS_HPP_

#include "zisa/memory/device_type.hpp"
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/add_arrays_interior_cuda.hpp>
#endif

template <int n_coupled, typename Scalar>
void add_arrays_interior_cpu(zisa::array_view<Scalar, 2> dst,
                             zisa::array_const_view<Scalar, 2> src,
                             Scalar scaling) {
  for (int i = 1; i < dst.shape(0) - 1; i++) {
    for (int j = n_coupled; j < dst.shape(1) - n_coupled; j++) {
      dst(i, j) += scaling * src(i, j);
    }
  }
}

// PRE: dimensions of src and dst match, both are stored on same device type
// POST: dst(i, j) = dst(i, j) + scaling * src(i, j) in interior
//       dst(i, j) = dst(i, j)                       on boundary
template <int n_coupled = 1, typename Scalar>
void add_arrays_interior(zisa::array_view<Scalar, 2> dst,
                         zisa::array_const_view<Scalar, 2> src,
                         Scalar scaling) {
  const zisa::device_type memory_dst = dst.memory_location();
  if (memory_dst != src.memory_location()) {
    std::cerr << "Error in add_arrays: dst and src have to have the same "
                 "memory location"
              << std::endl;
    exit(1);
  }
  if (dst.shape() != src.shape()) {
    std::cerr << "Error in add_arrays: dst and src have to have the same size"
              << std::endl;
    exit(1);
  }
  if (memory_dst == zisa::device_type::cpu) {
    add_arrays_interior_cpu<n_coupled>(dst, src, scaling);
  }
#if CUDA_AVAILABLE
  else if (memory_dst == zisa::device_type::cuda) {
    add_arrays_interior_cuda<n_coupled>(dst, src, scaling);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "Add arrays: Unknown device_type of inputs\n";
    exit(1);
  }
}

template <typename Scalar>
inline void print_matrix(const zisa::array_const_view<Scalar, 2> &array) {
#if CUDA_AVAILABLE
  zisa::array<Scalar, 2> cpu_data(array.shape());
  zisa::copy(cpu_data, array);
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      std::cout << cpu_data(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#else
  for (int i = 0; i < array.shape(0); i++) {
    for (int j = 0; j < array.shape(1); j++) {
      std::cout << array(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
#endif // CUDA_AVAILABLE
}

// make sure that the dimensions of the data to read and the preallocated
// data array match
template <typename Scalar>
inline void read_data(zisa::HierarchicalReader &reader,
                      zisa::array<Scalar, 2> &data, const std::string &tag) {
#if CUDA_AVAILABLE
  zisa::array<Scalar, 2> cpu_data(data.shape());
  zisa::load_impl<Scalar, 2>(reader, cpu_data, tag,
                             zisa::default_dispatch_tag{});
  zisa::copy(data, cpu_data);
#else
  zisa::load_impl(reader, data, tag, zisa::bool_dispatch_tag{});
#endif // CUDA_AVAILABLE
}

template <int n_coupled = 1, typename Scalar, typename Function>
void apply_function(zisa::array_view<Scalar, 2> dst,
                    zisa::array_const_view<Scalar, 2> src, Function &f) {
#if CUDA_AVAILABLE
  std::throw();
#else
  for (uint32_t i = 0; i < dst.shape(0) - 1; i++) {
    for (uint32_t j = 0; j < dst.shape(1) - n_coupled; j += n_coupled) {
      dst(i, j) = f(src(i, j));
    }
  }
#endif
}

#endif // HELPERS_HPP_
