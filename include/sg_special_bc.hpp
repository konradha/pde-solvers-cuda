#ifndef SG_SPECIAL_BC_HPP_
#define SG_SPECIAL_BC_HPP_
#include <zisa/memory/array.hpp>
// TODO CUDA implementation

template <int n_coupled, typename Scalar>
void special_sg_bc_cpu(
    zisa::array_view<Scalar, 2> &data, Scalar xL, Scalar xR, Scalar yT,
    Scalar yB,            // left, right, top, bottom
    Scalar dx, Scalar dy, // need measure of how much to enhance position
    Scalar t = 0.) {
  const unsigned x_length = data.shape(0);
  const unsigned y_length = data.shape(1);

  const unsigned x_shift = x_length - 2;
  const unsigned y_shift = y_length - 2 * n_coupled;

  // boundary condition taken from:
  // "Numerical solutions of a damped sine-Gordon equation in two space
  // variables" 1995
  auto f = [&](Scalar x, Scalar y, Scalar t) {
    return 4 * std::exp(x + y + t) /
           (std::exp(2 * t) + std::exp(2 * x + 2 * y));
  };

  auto left = [&](uint32_t yi, Scalar t) {
    Scalar y = yB + dy * yi;
    return f(xL, y, t);
  };

  auto right = [&](uint32_t yi, Scalar t) {
    Scalar y = yB + dy * yi;
    return f(xR, y, t);
  };

  auto top = [&](uint32_t xi, Scalar t) {
    Scalar x = xL + dx * xi;
    return f(x, yT, t);
  };

  auto bottom = [&](uint32_t xi, Scalar t) {
    Scalar x = xL + dx * xi;
    return f(x, yB, t);
  };

  assert(n_coupled == 1 && "These specific boundary conditions only work for a "
                           "single 2+1 d sine-Gordon");

  for (uint32_t yi = 0; yi < y_length; ++yi) {
    data(0, yi) = data(1, yi) - dy * left(yi, t);
  }

  for (uint32_t yi = 0; yi < y_length; ++yi) {
    data(x_length - 1, yi) = data(x_length - 2, yi) + dy * right(yi, t);
  }

  for (uint32_t xi = 1; xi < x_length - 1; ++xi) {
    data(xi, 0) = data(xi, 1) + dy * top(xi, t);
  }

  for (uint32_t xi = 1; xi < x_length - 1; ++xi) {
    data(xi, y_length - 1) = data(xi, y_length - 2) - dy * bottom(xi, t);
  }
}

template <int n_coupled, typename Scalar>
void special_sg_bc(zisa::array_view<Scalar, 2> data, Scalar dt, Scalar xL,
                   Scalar xR, Scalar yT, Scalar yB, Scalar dx, Scalar dy,
                   Scalar t = 0.) {
  const zisa::device_type memory_location = data.memory_location();
  if (memory_location == zisa::device_type::cpu) {
    special_sg_bc_cpu<n_coupled, Scalar>(data, xL, xR, yT, yB, dx, dy, t);
  }
#if CUDA_AVAILABLE
  else if (memory_location == zisa::device_type::cuda) {
    std::throw("CUDA version of special SG BCs not implemented yet!");
  }
#endif // CUDA_AVAILABLE
  else {
    std::cerr << "periodic bc unknown device_type of inputs\n";
  }
}

#endif // SG_SPECIAL_BC_HPP_
