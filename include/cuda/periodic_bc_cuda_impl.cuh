#ifndef PERIODIC_BC_CUDA_IMPL_H_
#define PERIODIC_BC_CUDA_IMPL_H_

#include <zisa/memory/array.hpp>

#ifndef THREAD_DIMS
#define THREAD_DIMS 1024
#endif

template <int n_coupled, typename Scalar>
__global__ void periodic_bc_cuda_kernel(zisa::array_view<Scalar, 2> data,
                                        unsigned data_size) {
  const unsigned idx = blockIdx.x * THREAD_DIMS + threadIdx.x;
  const unsigned x_length = data.shape(0);
  const unsigned y_length = data.shape(1);

  if (idx < data_size) {
    if (idx < y_length) {
      if (idx < n_coupled) {
        // upper left corner
        data(0, idx) = data(x_length - 2, y_length - 2 * n_coupled + idx);
      } else if (idx >= y_length - n_coupled) {
        // upper right corner
        data(0, idx) = data(x_length - 2, idx - y_length + 2 * n_coupled);
      } else {
        // upper boundary without corners
        data(0, idx) = data(x_length - 2, idx);
      }
      return;
    } else if (idx < y_length + (x_length - 2) * 2 * n_coupled) {
      const int loc_idx = idx - y_length;
      // left or right boundary
      if ((idx / n_coupled) % 2 == 0) {
        // left boundary
        const int x_idx = (loc_idx) / (2 * n_coupled) + 1;
        const int y_idx = loc_idx % n_coupled;
        data(x_idx, y_idx) = data(x_idx, y_length - 2 * n_coupled + y_idx);
      } else {
        const int x_idx = loc_idx / (2 * n_coupled) + 1;
        const int y_idx = loc_idx % n_coupled;
        data(x_idx, y_length - n_coupled + y_idx) =
            data(x_idx, y_idx + n_coupled);
      }
      return;
    } else {
      const int loc_idx = idx - (y_length + (x_length - 2) * 2 * n_coupled);
      if (loc_idx < n_coupled) {
        // lower left corner
        data(x_length - 1, loc_idx) =
            data(1, y_length - 2 * n_coupled + loc_idx);
      } else if (loc_idx >= y_length - n_coupled) {
        // lower right corner
        data(x_length - 1, loc_idx) = data(1, loc_idx + n_coupled - y_length);
      } else {
        data(x_length - 1, loc_idx) = data(1, loc_idx);
      }
      return;
    }
  }
}

template <int n_coupled, typename Scalar>
void periodic_bc_cuda(zisa::array_view<Scalar, 2> data) {
#if CUDA_AVAILABLE
  const unsigned thread_dims = THREAD_DIMS;
  // size of whole boundary where periodic bc has to be applied
  const unsigned data_size =
      data.shape(0) * n_coupled * 2 + (data.shape(1) - 2 * n_coupled) * 2;
  const unsigned block_dims = std::ceil((double)data_size / thread_dims);
  // std::cout << "should reach cuda " << block_dims << " "
  //           << "thread_dims" << std::endl;
  periodic_bc_cuda_kernel<n_coupled, Scalar>
      <<<block_dims, thread_dims>>>(data, data_size);
  const auto error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    std::cout << "Error in periodic_bc: " << cudaGetErrorString(error)
              << std::endl;
  }
#endif // CUDA_AVAILABLE
}
#endif // PERIODIC_BC_CUDA_H_
