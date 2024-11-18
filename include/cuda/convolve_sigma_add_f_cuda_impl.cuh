#ifndef CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
#define CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_

#ifndef NUM_THREAD_X
#define NUM_THREAD_X 16
#endif

#ifndef NUM_THREAD_Y
#define NUM_THREAD_Y 16
#endif

template <int n_coupled, typename Scalar, typename Function>
__global__ void
convolve_sigma_add_f_cuda_kernel(zisa::array_view<Scalar, 2> dst,
                                 zisa::array_const_view<Scalar, 2> src,
                                 zisa::array_const_view<Scalar, 2> sigma,
                                 Scalar del_x_2, Scalar del_y_2, Function f) {
  const int x_idx =
      threadIdx.x + 1 + blockIdx.x * NUM_THREAD_X; // cannot be on boundary
  const int y_idx = threadIdx.y + 1 + blockIdx.y * NUM_THREAD_Y;

  const int Nx = src.shape(0);
  const int Ny = src.shape(1) / n_coupled;
  if (x_idx < Nx - 1 && y_idx < Ny - 1) {
    Scalar result_function[n_coupled];
    f(zisa::array_const_view<Scalar, 1>{zisa::shape_t<1>(n_coupled),
                                        &src(x_idx, n_coupled * y_idx),
                                        zisa::device_type::cuda},
      result_function);
    for (int i = 0; i < n_coupled; i++) {
      dst(x_idx, n_coupled * y_idx + i) =
          del_x_2 * (sigma(2 * x_idx - 2, y_idx) *
                         (src(x_idx - 1, y_idx * n_coupled + i) -
                          src(x_idx, y_idx * n_coupled + i)) +
                     sigma(2 * x_idx, y_idx) *
                         (src(x_idx + 1, y_idx * n_coupled + i) -
                          src(x_idx, y_idx * n_coupled + i))) +
          del_y_2 * (sigma(2 * x_idx - 1, y_idx - 1) *
                         (src(x_idx, y_idx * n_coupled + i - n_coupled) -
                          src(x_idx, y_idx * n_coupled + i)) +
                     sigma(2 * x_idx - 1, y_idx + 1) *
                         (src(x_idx, y_idx * n_coupled + i + n_coupled) -
                          src(x_idx, y_idx * n_coupled + i))) +
          result_function[i];
    }
  }
}

template <int n_coupled, typename Scalar, typename Function>
void convolve_sigma_add_f_cuda(zisa::array_view<Scalar, 2> dst,
                               zisa::array_const_view<Scalar, 2> src,
                               zisa::array_const_view<Scalar, 2> sigma,
                               Scalar del_x_2, Scalar del_y_2,
                               const Function &f) {
#if CUDA_AVAILABLE
  const dim3 thread_dims(NUM_THREAD_X, NUM_THREAD_Y);
  const dim3 block_dims(
      std::ceil((src.shape(0) - 2) / (double)thread_dims.x),
      std::ceil((src.shape(1) - 2 * n_coupled) / (double)thread_dims.y) *
          n_coupled);
  convolve_sigma_add_f_cuda_kernel<n_coupled, Scalar, Function>
      <<<block_dims, thread_dims>>>(dst, src, sigma, del_x_2, del_y_2, f);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in convolve_sigma_add_f_cuda: "
              << cudaGetErrorString(error) << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // CONVOLVE_SIGMA_ADD_F_CUDA_IMPL_H_
