#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include <convolve_sigma_add_f.hpp>
#include <helpers.hpp>
#include <neumann_bc.hpp>
#include <periodic_bc.hpp>
#include <sg_special_bc.hpp>
#include <string>
#include <zisa/io/file_manipulation.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/io/hierarchical_file.hpp>
#include <zisa/io/hierarchical_reader.hpp>
#include <zisa/io/netcdf_file.hpp>
#include <zisa/io/netcdf_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/array_traits.hpp>
#include <zisa/memory/device_type.hpp>
#include <zisa/memory/memory_location.hpp>
#include <zisa/memory/shape.hpp>

#define PROGRESS_BAR_WIDTH 100

#define PRINT_PROGRESS(iteration, total)                                       \
  do {                                                                         \
    static int last_printed = -1;                                              \
    static clock_t last_update = 0;                                            \
    clock_t now = clock();                                                     \
    if (iteration == 0) {                                                      \
      last_update = now;                                                       \
    }                                                                          \
    int percentage = (iteration * 100) / total;                                \
    if (percentage != last_printed &&                                          \
        (now - last_update) > CLOCKS_PER_SEC / 10) {                           \
      int filled_width = (PROGRESS_BAR_WIDTH * iteration) / total;             \
      printf("\r[");                                                           \
      for (int i = 0; i < PROGRESS_BAR_WIDTH; ++i) {                           \
        if (i < filled_width)                                                  \
          printf("#");                                                         \
        else                                                                   \
          printf(" ");                                                         \
      }                                                                        \
      printf("] %3d%%", percentage);                                           \
      fflush(stdout);                                                          \
      last_printed = percentage;                                               \
      last_update = now;                                                       \
    }                                                                          \
    if (iteration == total - 1) {                                              \
      printf("\n");                                                            \
    }                                                                          \
  } while (0)

#define DURATION(a)                                                            \
  std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define NOW std::chrono::high_resolution_clock::now()

enum BoundaryCondition { Dirichlet, Neumann, Periodic, SpecialSG };
// SGSpecial is:
// u_x(x, y, t) = -4 exp(x + y + t) / (1 + exp(2x + 2y)) for x = L and x = -L
// u_y(x, y, t) = -4 exp(x + y + t) / (1 + exp(2x + 2y)) for y = L and y = -L

template <int n_coupled, typename Scalar> class PDEBase {
public:
  // note here that Nx and Ny denote the size INSIDE the boundary WITHOUT the
  // boundary so that the total size is (Nx + 2) * (Ny + 2)
  // if n_coupled > 1, the coupled values are saved next to each other in data_
  // and bc_neumann_values_, for example for n_coupled = 3, data_ = u(0, 0),
  // v(0, 0), w(0, 0), u(0, 1), v(0, 1)...
  //         u(1, 0), v(1, 0), w(1, 0), u(1, 1), v(1, 1)...
  // note that sigma_values is independent of n_coupled

  // TODO(konradha) check if we can optimize data layout here
  // to get faster cache mechanics
  PDEBase(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Scalar dx, Scalar dy)
      : data_(zisa::shape_t<2>(Nx + 2, n_coupled * (Ny + 2)), memory_location),
        bc_neumann_values_(zisa::shape_t<2>(Nx + 2, n_coupled * (Ny + 2)),
                           memory_location),
        sigma_values_(zisa::shape_t<2>(2 * Nx + 1, Ny + 1), memory_location),
        memory_location_(memory_location), bc_(bc), dx_(dx), dy_(dy) {}

  PDEBase(const PDEBase &other)
      : data_(other.data_.shape(), other.memory_location_),
        bc_neumann_values_(other.bc_neumann_values_.shape(),
                           other.memory_location_),
        sigma_values_(other.sigma_values_.shape(), other.memory_location_),
        memory_location_(other.memory_location_), bc_(other.bc_),
        dx_(other.dx_), dy_(other.dy_) {
    zisa::copy(data_, other.data_);
    zisa::copy(bc_neumann_values_, other.bc_neumann_values_);
    zisa::copy(sigma_values_, other.sigma_values_);
  }

  virtual void apply(Scalar dt) = 0;

  // apply timesteps and save snapshots at times T/n_snapshots
  // note that for this we sometimes have to change the timestep
  template <typename WRITER>
  void apply_with_snapshots(Scalar T, unsigned int n_timesteps,
                            unsigned int n_snapshots, WRITER &writer,
                            int n_member = 0) {

    Scalar dt = T / n_timesteps;
    Scalar time = 0.;
    unsigned int snapshot_counter = 0;
    Scalar dsnapshots = T / (n_snapshots - 1);
    // save initial data
    writer.save_snapshot(n_member, snapshot_counter, data_.const_view());
    snapshot_counter++;
    Scalar total_comp_time_count = 0.;
    int tot_comp_count = 0;
    for (unsigned int i = 0; i < n_timesteps; ++i) {
      if (time + dt >= dsnapshots * snapshot_counter) {
        Scalar dt_new = dsnapshots * snapshot_counter - time;
        apply(dt_new);

        writer.save_snapshot(n_member, snapshot_counter, data_.const_view());
        apply(dt - dt_new);
        snapshot_counter++;
      } else {
        apply(dt);
        // print();
        // throw std::logic_error("Gracefully shutting down");
      }
      if (memory_location_ == zisa::device_type::cpu)
        PRINT_PROGRESS(i, n_timesteps);
      time += dt;
    }

    if (snapshot_counter < n_snapshots) {
      // total Time doesn't reach T due to numerical errors. Add more timesteps
      Scalar dt_new = T - time;
      // auto start = NOW;
      apply(dt_new);
      // auto end = NOW;
      // total_comp_time_count += DURATION(end - start);
      // tot_comp_count++;
      writer.save_snapshot(n_member, snapshot_counter, data_.const_view());
    }
    // std::cout << "number steps: " << tot_comp_count << std::endl;
    // std::cout << "total time: " << total_comp_time_count << " micros" <<
    // std::endl; std::cout << "avg time per step: " << total_comp_time_count /
    // tot_comp_count << " micros" << std::endl;
  }

  // for testing, does this work if on gpu?
  zisa::array_const_view<Scalar, 2> get_data() { return data_.const_view(); }

  zisa::array_const_view<Scalar, 2> get_sigma() {
    return sigma_values_.const_view();
  }

  zisa::array_const_view<Scalar, 2> get_bc() {
    return bc_neumann_values_.const_view();
  }

  BoundaryCondition get_bc_type() { return bc_; }

  // for testing/debugging
  void print() {
    std::cout << "data has size x: " << data_.shape(0)
              << ", y: " << data_.shape(1) << std::endl;

    std::cout << "data:" << std::endl;
    print_matrix(data_.const_view());
    // do not print bc and sigma
    // return;
    std::cout << "bc values:" << std::endl;
    print_matrix(bc_neumann_values_.const_view());
    std::cout << "sigma values:" << std::endl;
    print_matrix(sigma_values_.const_view());
  }

protected:
  // apply boundary conditions
  // for cuda implementation, this should probably be done in the same step as
  // applying the convolution to avoid copying data back and forth
  void add_bc(Scalar dt, Scalar xL = 0., Scalar xR = 0., Scalar yT = 0.,
              Scalar yB = 0., Scalar dx = 0., Scalar dy = 0., Scalar t = 0.) {
    if (bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
      // dirichlet_bc(data_.view(), bc_neumann_values_.const_view());
    } else if (bc_ == BoundaryCondition::Neumann) {
      neumann_bc<n_coupled, Scalar>(data_.view(),
                                    bc_neumann_values_.const_view(), dt);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(data_.view());
    } else if (bc_ == BoundaryCondition::SpecialSG) {
#if DEBUG
      std::cout << "applying special boundary condition\n";
      std::cout << "xL: " << xL << "\n";
      std::cout << "xR: " << xR << "\n";
      std::cout << "yB: " << yB << "\n";
      std::cout << "yT: " << yT << "\n";
      std::cout << "dx: " << dx << "\n";
      std::cout << "dy: " << dy << "\n";
      std::cout << "t:  " << t << "\n";
#endif
      special_sg_bc<n_coupled, Scalar>(data_.view(), dt, xL, xR, yT, yB, dx, dy,
                                       t);
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  zisa::array<Scalar, 2> data_;
  zisa::array<Scalar, 2> bc_neumann_values_;
  zisa::array<Scalar, 2> sigma_values_;

  const BoundaryCondition bc_;
  const zisa::device_type memory_location_;

  const Scalar dx_;
  const Scalar dy_;
  bool ready_ = false;
};
#undef DURATION

#endif // PDE_BASE_HPP_
