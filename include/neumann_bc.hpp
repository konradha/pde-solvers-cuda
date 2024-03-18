#ifndef NEUMANN_BC_HPP_
#define NEUMANN_BC_HPP_
#include <zisa/memory/array.hpp>
#if CUDA_AVAILABLE
#include <cuda/neumann_bc_cuda.hpp>
#endif

template <typename Scalar>
void neumann_bc_cpu(zisa::array_view<Scalar, 2> data,
                    const zisa::array_const_view<Scalar, 2> &bc,
                    unsigned n_ghost_cells_x,
                    unsigned n_ghost_cells_y, double dt) {
  unsigned x_length = data.shape(0);
  unsigned y_length = data.shape(1);


  for (int x_idx = n_ghost_cells_x; x_idx < x_length - n_ghost_cells_x; x_idx++) {
    // left cols without corners
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) += dt * bc(x_idx, y_idx);
    }
    // right cols without corners
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_length - 1 - y_idx) += dt * bc(x_idx, y_length - 1 - y_idx);
    }
  }

  for (int y_idx = n_ghost_cells_y; y_idx < y_length - n_ghost_cells_y; y_idx++) {
    // top cols without corners
    for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
      data(x_idx, y_idx) += dt * bc(x_idx, y_idx);
    }

    // bottom cols without corners
    for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
      data(x_length - x_idx - 1, y_idx) += dt * bc(x_length - x_idx - 1, y_idx);
    }
  }
  // corners
  for (int x_idx = 0; x_idx < n_ghost_cells_x; x_idx++) {
    for (int y_idx = 0; y_idx < n_ghost_cells_y; y_idx++) {
      data(x_idx, y_idx) += dt * bc(x_idx, y_idx);
      data(x_length - x_idx - 1, y_idx) += dt * bc(x_length - x_idx - 1, y_idx);
      data(x_idx, y_length - y_idx - 1) += dt * bc(x_idx, y_length - y_idx - 1);
      data(x_length - x_idx - 1, y_length - y_idx - 1) += dt * bc(x_length - x_idx - 1, y_length - y_idx  -1);
    }
  }
}

// updates the outermost data with dt * values on bc
template <typename Scalar>
void neumann_bc(zisa::array_view<Scalar, 2> data,
                const zisa::array_const_view<Scalar, 2> &bc,
                unsigned n_ghost_cells_x,
                unsigned n_ghost_cells_y, zisa::device_type memory_location, double dt) {
  if (memory_location == zisa::device_type::cpu) {
    neumann_bc_cpu(data, bc, n_ghost_cells_x, n_ghost_cells_y, dt);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    // TODO
    neumann_bc_cuda(data, bc, n_ghost_cells_x, n_ghost_cells_y, dt);
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "neumann bc unknown device_type of inputs\n";
  }
}

#endif // NEUMANN_BC_HPP_
