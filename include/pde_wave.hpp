#ifndef PDE_WAVE_HPP_
#define PDE_WAVE_HPP_

#include "io/netcdf_reader.hpp"
#include "periodic_bc.hpp"
#include "zisa/io/hdf5_writer.hpp"
#include <pde_base.hpp>

#include <cmath>
#include <stdexcept>

template <int n_coupled, typename Scalar, typename Function>
class PDEWave : public virtual PDEBase<n_coupled, Scalar> {
public:
  PDEWave(unsigned Nx, unsigned Ny, const zisa::device_type memory_location,
          BoundaryCondition bc, Function f, Scalar dx, Scalar dy)
      : PDEBase<n_coupled, Scalar>(Nx, Ny, memory_location, bc, dx, dy),
        func_(f) {}

  void apply(Scalar dt) override {
    if (!this->ready_) {
      std::cerr << "Wave solver is not ready yet! Read data first" << std::endl;
      return;
    }

    zisa::array<Scalar, 2> second_deriv(this->data_.shape(),
                                        this->data_.device());
    const Scalar del_x_2 = 1. / (this->dx_ * this->dx_);
    const Scalar del_y_2 = 1. / (this->dy_ * this->dy_);
    convolve_sigma_add_f<n_coupled>(
        second_deriv.view(), this->data_.const_view(),
        this->sigma_values_.const_view(), del_x_2, del_y_2, func_);

    // update of derivative
    add_arrays_interior<n_coupled>(this->bc_neumann_values_.view(),
                                   second_deriv.const_view(), dt);

    if (this->bc_ != BoundaryCondition::SpecialSG) {
      // update of data
      add_arrays_interior<n_coupled>(this->data_.view(),
                                     this->bc_neumann_values_.const_view(), dt);
      PDEBase<n_coupled, Scalar>::add_bc(dt);
    } else {
      // assuming really basic geometry: [0, xR] x [0, yT] -- might be changed
      // in future work
      const auto Nx = this->data_.shape(0);
      const auto Ny = this->data_.shape(1);
      Scalar xL = 0.;
      Scalar yB = 0.;

      Scalar xR = this->dx_ * Nx;
      Scalar yT = this->dy_ * Ny;
      PDEBase<n_coupled, Scalar>::add_bc(dt, xL, xR, yT, yB, this->dx_,
                                         this->dy_, this->current_t_);
    }
    this->current_t_ += dt;
  }

  void read_values(const std::string &filename,
                   const std::string &tag_data = "initial_data",
                   const std::string &tag_sigma = "sigma",
                   const std::string &tag_initial_derivative = "init_deriv") {
    zisa::HDF5SerialReader reader(filename);

    read_data(reader, this->data_, tag_data);
    read_data(reader, this->sigma_values_, tag_sigma);
    read_data(reader, this->bc_neumann_values_, tag_initial_derivative);

    if (this->bc_ == BoundaryCondition::Neumann) {
      zisa::copy(this->bc_neumann_values_, this->bc_neumann_values_);
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    } else if (this->bc_ == BoundaryCondition::SpecialSG) {
      throw std::logic_error("Not implemented yet 1");
    }
    this->ready_ = true;
  }

  void read_values(zisa::array_const_view<Scalar, 2> data,
                   zisa::array_const_view<Scalar, 2> sigma,
                   zisa::array_const_view<Scalar, 2> bc,
                   zisa::array_const_view<Scalar, 2> initial_derivative) {
    zisa::copy(this->data_, data);
    zisa::copy(this->sigma_values_, sigma);
    zisa::copy(this->bc_neumann_values_, initial_derivative);

    if (this->bc_ == BoundaryCondition::Neumann) {
      zisa::copy(this->bc_neumann_values_, initial_derivative);
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    } else if (this->bc_ == BoundaryCondition::SpecialSG) {
      throw std::logic_error("Not implemented yet 2");
    }
    this->ready_ = true;
  }

  void read_initial_data_from_netcdf(const NetCDFPDEReader &reader, int memb) {
#if CUDA_AVAILABLE
    zisa::array<Scalar, 2> tmp(
        zisa::shape_t<2>(this->data_.shape()[0], this->data_.shape()[1]),
        zisa::device_type::cpu);
    reader.write_variable_of_member_to_array("initial_data", tmp.view().raw(),
                                             memb, this->data_.shape()[0],
                                             this->data_.shape()[1]);
    zisa::copy(this->data_, tmp);
#else
    reader.write_variable_of_member_to_array(
        "initial_data", this->data_.view().raw(), memb, this->data_.shape()[0],
        this->data_.shape()[1]);
#endif

#if CUDA_AVAILABLE
    zisa::array<Scalar, 2> tmp_sigma(
        zisa::shape_t<2>(this->sigma_values_.shape()[0],
                         this->sigma_values_.shape()[1]),
        zisa::device_type::cpu);
    reader.write_variable_of_member_to_array(
        "sigma_values", tmp_sigma.view().raw(), memb,
        this->sigma_values_.shape()[0], this->sigma_values_.shape()[1]);
    zisa::copy(this->sigma_values_, tmp_sigma);

#else
    reader.write_variable_of_member_to_array(
        "sigma_values", this->sigma_values_.view().raw(), memb,
        this->sigma_values_.shape()[0], this->sigma_values_.shape()[1]);
#endif

#if CUDA_AVAILABLE
    reader.write_variable_of_member_to_array(
        "bc_neumann_values", tmp.view().raw(), memb,
        this->bc_neumann_values_.shape()[0],
        this->bc_neumann_values_.shape()[1]);
    zisa::copy(this->bc_neumann_values_, tmp);
#else
    reader.write_variable_of_member_to_array(
        "bc_neumann_values", this->bc_neumann_values_.view().raw(), memb,
        this->bc_neumann_values_.shape()[0],
        this->bc_neumann_values_.shape()[1]);
#endif

    if (this->bc_ == BoundaryCondition::Neumann) {
      // do nothing as bc_data is already loaded
    } else if (this->bc_ == BoundaryCondition::Dirichlet) {
      // do nothing as long as data on boundary does not change
    } else if (this->bc_ == BoundaryCondition::Periodic) {
      periodic_bc<n_coupled, Scalar>(this->data_.view());
    } else if (this->bc_ == BoundaryCondition::SpecialSG) {
      // do nothing, initial data is loaded
    }

    // std::cout << reader.get_extra_source_term() << "\n";
    // std::cout << reader.get_extra_source_eps() << "\n";

    // update function scalings
    func_.update_values(reader.get_function<Scalar>(memb).const_view());

    this->ready_ = true;
  }
  void print_deriv() {
    std::cout << "deriv: " << std::endl;
    print_matrix(this->bc_neumann_values_.const_view());
  }
  void print_func() {
    std::cout << "function: " << std::endl;
    func_.print();
  }

protected:
  Function func_;
  Scalar current_t_ = 0.;
};

#endif // PDE_WAVE_HPP_
