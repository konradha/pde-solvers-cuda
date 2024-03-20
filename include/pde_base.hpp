#ifndef PDE_BASE_HPP_
#define PDE_BASE_HPP_

#include "zisa/io/file_manipulation.hpp"
#include "zisa/memory/array_traits.hpp"
#include "zisa/memory/memory_location.hpp"
#include "zisa/memory/shape.hpp"
#include <convolution.hpp>
#include <dirichlet_bc.hpp>
#include <neumann_bc.hpp>
#include <periodic_bc.hpp>
#include <zisa/io/hdf5_serial_writer.hpp>
#include <zisa/memory/array.hpp>
#include <zisa/memory/device_type.hpp>

template <typename Scalar, typename BoundaryCondition> class PDEBase {
public:
  using scalar_t = Scalar;

  PDEBase(unsigned Nx, unsigned Ny,
          const zisa::array_const_view<Scalar, 2> &kernel, BoundaryCondition bc)
      : data_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2),
                               Ny + 2 * (kernel.shape(1) / 2)),
              kernel.memory_location()),
        bc_values_(zisa::shape_t<2>(Nx + 2 * (kernel.shape(0) / 2),
                                    Ny + 2 * (kernel.shape(1) / 2)),
                   kernel.memory_location()),
        sigma_values_(zisa::shape_t<2>(Nx + 1, Ny + 1), kernel.memory_location()),
        kernel_(kernel), bc_(bc) {}


  void read_values(const std::string &filename) {
    zisa::HDF5SerialReader reader(filename);
    zisa::load_impl<Scalar, 2>(reader, data_, "initial_data", zisa::default_dispatch_tag{});
    if (bc_ == BoundaryCondition::Neumann) {
      zisa::load_impl(reader, bc_values_, "bc", zisa::bool_dispatch_tag{});
    } else if (bc_ == BoundaryCondition::Dirichlet) {
      zisa::copy(bc_values_, data_);
    } else if (bc_ == BoundaryCondition::Periodic) {
      add_bc();
    }
    zisa::load_impl(reader, sigma_values_, "sigma", zisa::bool_dispatch_tag{});
    ready_ = true;
    std::cout << "initial data, sigma and boundary conditions read!" << std::endl;
    print();
  }

  void apply() {
    zisa::array<scalar_t, 2> tmp(data_.shape(), data_.device());
    convolve(tmp.view(), data_.const_view(), this->kernel_);
    if (bc_ == BoundaryCondition::Neumann) {
      // make shure that boundary values stay constant to later apply boundary
      // conditions (they where not copied in convolve)
      dirichlet_bc(tmp.view(), data_.const_view(), num_ghost_cells_x(),
                   num_ghost_cells_y(), kernel_.memory_location());
    }
    zisa::copy(data_, tmp);
    add_bc();
  }

  unsigned num_ghost_cells(unsigned dir) { return kernel_.shape(dir) / 2; }
  unsigned num_ghost_cells_x() { return num_ghost_cells(0); }
  unsigned num_ghost_cells_y() { return num_ghost_cells(1); }

  // for testing/debugging
  void print() {
    int x_size = data_.shape(0);
    int y_size = data_.shape(1);
    std::cout << "data has size x: " << x_size << ", y: " << y_size
              << std::endl;
    std::cout << "border sizes are x: " << num_ghost_cells_x()
              << ", y: " << num_ghost_cells_y() << std::endl;
// weird segmentation fault if using cuda
// how is it possible to print an array on gpus?
#if CUDA_AVAILABLE
    zisa::array<float, 2> cpu_data(zisa::shape_t<2>(x_size, y_size));
    zisa::copy(cpu_data, data_);
    for (int i = 0; i < x_size; i++) {
      for (int j = 0; j < y_size; j++) {
        std::cout << cpu_data(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
#endif
    for (int i = 0; i < x_size; i++) {
      for (int j = 0; j < y_size; j++) {
        std::cout << data_(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    
// weird segmentation fault if using cuda
// how is it possible to print an array on gpus?
#if CUDA_AVAILABLE
    zisa::array<float, 2> cpu_bc(zisa::shape_t<2>(x_size, y_size));
    zisa::copy(cpu_bc, bc_values_);
    for (int i = 0; i < x_size; i++) {
      for (int j = 0; j < y_size; j++) {
        std::cout << cpu_bc(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
#endif
    for (int i = 0; i < x_size; i++) {
      for (int j = 0; j < y_size; j++) {
        std::cout << bc_values_(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  

// weird segmentation fault if using cuda
// how is it possible to print an array on gpus?
#if CUDA_AVAILABLE
    zisa::array<float, 2> cpu_sigma(zisa::shape_t<2>(x_size - 1, y_size - 1));
    zisa::copy(cpu_sigma, sigma_values_);
    for (int i = 0; i < x_size - 1; i++) {
      for (int j = 0; j < y_size - 1; j++) {
        std::cout << cpu_sigma(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    return;
#endif
    for (int i = 0; i < x_size - 1; i++) {
      for (int j = 0; j < y_size - 1; j++) {
        std::cout << sigma_values_(i, j) << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

protected:
  void add_bc() {
    if (bc_ == BoundaryCondition::Dirichlet) {
      dirichlet_bc<Scalar>(data_.view(), bc_values_.const_view(),
                           num_ghost_cells_x(), num_ghost_cells_y(),
                           kernel_.memory_location());
    } else if (bc_ == BoundaryCondition::Neumann) {
      // TODO: change dt
      neumann_bc(data_.view(), bc_values_.const_view(), num_ghost_cells_x(),
                 num_ghost_cells_y(), kernel_.memory_location(), 0.1);
    } else if (bc_ == BoundaryCondition::Periodic) {
      periodic_bc(data_.view(), num_ghost_cells_x(), num_ghost_cells_y(),
                  kernel_.memory_location());
    } else {
      std::cout << "boundary condition not implemented yet!" << std::endl;
    }
  }

  // void read_and_store_file(const std::string &filename,
  //                          const std::string &group_name,
  //                          const std::string &tag,
  //                          zisa::array<Scalar, 2> data_location, unsigned Nx,
  //                          unsigned Ny) {
  //   zisa::HDF5SerialReader serial_reader(filename);
  //   // zisa::load_impl(serial_reader, data_location, tag, zisa::default_dispatch_tag{});
  //   Scalar return_data[Nx][Ny];

  //   serial_reader.open_group(group_name);
  //   serial_reader.read_array(return_data, zisa::erase_data_type<Scalar>(), tag);

  //   // TODO: Optimize
  //   if (kernel_.memory_location() == zisa::device_type::cpu) {
  //     for (int i = 0; i < Nx; i++) {
  //       for (int j = 0; j < Ny; j++) {
  //         data_location(i, j) = return_data[i][j];
  //       }
  //     }
  //   } else if (kernel_.memory_location() == zisa::device_type::cuda) {
  //     zisa::array<Scalar, 2> tmp(
  //         zisa::shape_t<2>(Nx, Ny),
  //         zisa::device_type::cpu);
  //     for (int i = 0; i < Nx; i++) {
  //       for (int j = 0; j < Ny; j++) {
  //         tmp(i, j) = return_data[i][j];
  //       }
  //     }
  //     zisa::copy(data_location, tmp);
  //   } else {
  //     std::cout << "only data on cpu and cuda supported" << std::endl;
  //   }
  //   add_bc();
  // }

  zisa::array<Scalar, 2> data_;
  const zisa::array_const_view<Scalar, 2> kernel_;
  const BoundaryCondition bc_;
  zisa::array<Scalar, 2> bc_values_;
  zisa::array<Scalar, 2> sigma_values_;
  bool ready_ = false;
};

#endif // PDE_BASE_HPP_
