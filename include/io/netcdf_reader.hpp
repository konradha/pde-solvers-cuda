#ifndef NETCDF_READER_HPP_
#define NETCDF_READER_HPP_

#include "coupled_function.hpp"
#include "zisa/memory/array_decl.hpp"
#include <condition_variable>
#include <iostream>
#include <netcdf.h>
#include <string>

inline void check(int stat) {
  if (stat != NC_NOERR) {
    printf("NetCDF error: %s\n", nc_strerror(stat));
    exit(1);
  }
}

// note that this reader can only write to arrays on the cpu!
class NetCDFPDEReader {
public:
  NetCDFPDEReader(std::string filename) : filename_(filename) {
    check(nc_open(filename.c_str(), NC_NOWRITE, &ncid_));

    check(nc_get_att(ncid_, NC_GLOBAL, "scalar_type", &scalar_type_));

    check(nc_get_att(ncid_, NC_GLOBAL, "type_of_equation", &type_of_equation_));
    check(nc_get_att(ncid_, NC_GLOBAL, "n_members", &n_members_));

    check(nc_get_att(ncid_, NC_GLOBAL, "x_size", &x_size_));
    check(nc_get_att(ncid_, NC_GLOBAL, "x_length", &x_length_));
    check(nc_get_att(ncid_, NC_GLOBAL, "y_size", &y_size_));
    check(nc_get_att(ncid_, NC_GLOBAL, "y_length", &y_length_));

    check(nc_get_att(ncid_, NC_GLOBAL, "boundary_value_type",
                     &boundary_value_type_));
    // ordering matters for this reader!
    check(
        nc_get_att(ncid_, NC_GLOBAL, "extra_source_term", &extra_source_term_));

    check(nc_get_att(ncid_, NC_GLOBAL, "n_coupled", &n_coupled_));
    check(nc_get_att(ncid_, NC_GLOBAL, "coupled_function_order",
                     &coupled_function_order_));

    check(nc_get_att(ncid_, NC_GLOBAL, "number_timesteps", &number_timesteps_));
    check(nc_get_att(ncid_, NC_GLOBAL, "final_time", &final_time_));

    check(nc_get_att(ncid_, NC_GLOBAL, "number_snapshots", &number_snapshots_));

    size_t file_to_save_output_len = 0;
    check(nc_inq_attlen(ncid_, NC_GLOBAL, "file_to_save_output",
                        &file_to_save_output_len));

    file_to_save_output_ =
        (char *)malloc(file_to_save_output_len * sizeof(char));
    check(nc_get_att_text(ncid_, NC_GLOBAL, "file_to_save_output",
                          file_to_save_output_));
  }
  NetCDFPDEReader() = delete;
  NetCDFPDEReader(const NetCDFPDEReader &other) = delete;
  ~NetCDFPDEReader() {
    if (nc_close(ncid_) != NC_NOERR) {
      std::cout << "error in closing netcdf input file" << std::endl;
    }
    free(file_to_save_output_);
  }

  // getters
  // 0->HeatEquation, 1->WaveEquation
  int get_equation_type() const { return type_of_equation_; }
  int get_n_members() const { return n_members_; }

  int get_x_size() const { return x_size_; }
  double get_x_length() const { return x_length_; }
  int get_y_size() const { return y_size_; }
  double get_y_length() const { return y_length_; }

  // 0->Dirichlet, 1->Neumann, 2->Periodic
  int get_boundary_value() const { return boundary_value_type_; }
  // 0->float, 1->double
  int get_scalar_type() const { return scalar_type_; }

  int get_n_coupled() const { return n_coupled_; }
  int get_coupled_function_order() const { return coupled_function_order_; }

  int get_number_timesteps() const { return number_timesteps_; }
  double get_final_time() const { return final_time_; }
  int get_number_snapshots() const { return number_snapshots_; }

  int get_extra_source_term() const { return extra_source_term_; }

  char *get_file_to_save_output() const { return file_to_save_output_; }

  void write_whole_variable_to_array(std::string varname, void *arr_ptr) const {
    if (scalar_type_ == 0) {
      write_variable_to_array_generic(varname, (float *)arr_ptr);
    } else if (scalar_type_ == 1) {
      write_variable_to_array_generic(varname, (double *)arr_ptr);
    } else {
      std::cout
          << "error in write variable, scalar_type_ is not defined properly"
          << std::endl;
    }
  }

  void write_variable_of_member_to_array(std::string varname, void *arr_ptr,
                                         size_t member, size_t chunksize_x,
                                         size_t chunksize_y) const {
    if (scalar_type_ == 0) {
      write_variable_of_member_to_array_generic(
          varname, (float *)arr_ptr, member, chunksize_x, chunksize_y);
    } else if (scalar_type_ == 1) {
      write_variable_of_member_to_array_generic(
          varname, (double *)arr_ptr, member, chunksize_x, chunksize_y);
    } else {
      std::cout
          << "error in write variable, scalar_type_ is not defined properly"
          << std::endl;
      exit(-1);
    }
  }

  template <typename Scalar>
  zisa::array<Scalar, 1> get_function(size_t member) const {
    size_t len =
        n_coupled_ * (size_t)std::pow(coupled_function_order_, n_coupled_);
    zisa::array<Scalar, 1> scalings(zisa::shape_t<1>(len),
                                    zisa::device_type::cpu);

    int varid;
    check(nc_inq_varid(ncid_, "function_scalings", &varid));
    // TODO: this could be faster, for example use nc_get_var_float
    size_t startp[3] = {member, 0};
    size_t countp[3] = {1, len};
    check(nc_get_vara(ncid_, varid, startp, countp, scalings.view().raw()));
    return scalings;
  }

private:
  template <typename Scalar>
  void write_variable_to_array_generic(std::string varname,
                                       Scalar *arr_ptr) const {

    int varid;
    check(nc_inq_varid(ncid_, varname.c_str(), &varid));
    check(nc_get_var(ncid_, varid, arr_ptr));
  }

  template <typename Scalar>
  void write_variable_of_member_to_array_generic(std::string varname,
                                                 Scalar *arr_ptr, size_t member,
                                                 size_t chunksize_x,
                                                 size_t chunksize_y) const {

    int varid;
    check(nc_inq_varid(ncid_, varname.c_str(), &varid));
    // TODO: this could be faster, for example use nc_get_var_float
    size_t startp[3] = {member, 0, 0};
    size_t countp[3] = {1, chunksize_x, chunksize_y};
    check(nc_get_vara(ncid_, varid, startp, countp, arr_ptr));
  }

  std::string filename_;
  int ncid_;

  int type_of_equation_;
  int n_members_;

  int x_size_;
  int y_size_;
  double x_length_;
  double y_length_;

  int boundary_value_type_; // 0->Dirichlet, 1->Neumann, 2->PeriodicDirichlet,
  int scalar_type_;         // 0->float, 1->double

  int extra_source_term_;

  int n_coupled_;
  int coupled_function_order_;

  int number_timesteps_;
  double final_time_;
  int number_snapshots_;

  char *file_to_save_output_;
};

#endif // NETCDF_READER_HPP_
