#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <typename Scalar, typename Function>
__global__ void convolve_sigma_add_f_cuda_kernel(
    zisa::array_view<Scalar, 2> dst, zisa::array_const_view<Scalar, 2> src,
    zisa::array_const_view<Scalar, 2> sigma, Scalar del_x_2, Function f) {
  const int linear_idx = threadIdx.x + blockIdx.x * THREAD_DIMS;
  const int Nx = src.shape(0) - 2;
  const int Ny = src.shape(1) - 2;
  if (linear_idx < Nx * Ny) {
    const int x_idx = 1 + linear_idx / Ny;
    const int y_idx = 1 + linear_idx % Ny;
    dst(x_idx, y_idx) =
        del_x_2 *
            (sigma(2 * x_idx - 1, y_idx - 1) * src(x_idx, y_idx - 1) +
             sigma(2 * x_idx - 1, y_idx) * src(x_idx, y_idx + 1) +
             sigma(2 * x_idx - 2, y_idx - 1) * src(x_idx - 1, y_idx) +
             sigma(2 * x_idx, y_idx - 1) * src(x_idx + 1, y_idx) -
             (sigma(2 * x_idx - 1, y_idx - 1) + sigma(2 * x_idx - 1, y_idx) +
              sigma(2 * x_idx - 2, y_idx - 1) + sigma(2 * x_idx, y_idx - 1)) *
                 src(x_idx, y_idx)) +
        f(src(x_idx, y_idx));
  }
}

template <typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Function f) {
#if CUDA_AVAILABLE
  const int thread_dims = THREAD_DIMS;
  const int block_dims = std::ceil(
      (double)((src.shape(0) - 2) * (src.shape(1) - 2)) / thread_dims);
  std::cout << "conv_sigma on cuda not implemented yet" << std::endl;
  convolve_sigma_add_f_cuda_kernel<<<block_dims, thread_dims>>>(dst, src, sigma,
                                                                del_x_2, f);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_cuda: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
