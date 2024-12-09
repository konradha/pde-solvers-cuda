#ifndef RUN_FROM_NETCDF_HPP
#define RUN_FROM_NETCDF_HPP

#include "io/netcdf_reader.hpp"
#include "io/netcdf_writer.hpp"
#include "pde_base.hpp"
#include <chrono>
#include <coupled_function.hpp>
#include <iostream>
#include <netcdf.h>
#include <pde_heat.hpp>
#include <pde_wave.hpp>
#include <string.h>
#include <string>
#include <zisa/memory/array.hpp>

#define DURATION(a)                                                            \
  std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define NOW std::chrono::high_resolution_clock::now()

#define INSTANCIATE_PDE_AND_CALCULATE(PDE_TYPE, N_COUPLED, MEMORY_LOCATION)    \
  case N_COUPLED:                                                              \
    calculate_and_save_snapshots<                                              \
        PDE_TYPE<N_COUPLED, Scalar, CoupledFunction<Scalar>>, Scalar>(         \
        std::move(PDE_TYPE<N_COUPLED, Scalar, CoupledFunction<Scalar>>(        \
            reader.get_x_size(), reader.get_y_size(), MEMORY_LOCATION, bc,     \
            func_coupled, reader.get_x_length() / reader.get_x_size(),         \
            reader.get_x_length() / reader.get_x_size())),                     \
        reader);                                                               \
    break;

// read initial data, apply steps and save them to a output file
template <typename PDE, typename Scalar>
inline void calculate_and_save_snapshots(PDE pde,
                                         const NetCDFPDEReader &reader) {

  reader.get_number_snapshots();
  reader.get_file_to_save_output();

  NetCDFPDEWriter<Scalar> writer(
      reader.get_number_snapshots(), reader.get_final_time(),
      reader.get_n_members(), reader.get_x_size(), reader.get_x_length(),
      reader.get_y_size(), reader.get_y_length(), reader.get_equation_type(),
      reader.get_boundary_value(), reader.get_n_coupled(),
      reader.get_coupled_function_order(), reader.get_number_snapshots(),
      reader.get_file_to_save_output());
  for (int memb = 0; memb < reader.get_n_members(); memb++) {
    pde.read_initial_data_from_netcdf(reader, memb);
    // pde.print();
    // pde.print_func();
    // auto start = NOW;
    pde.apply_with_snapshots(reader.get_final_time(),
                             reader.get_number_timesteps(),
                             reader.get_number_snapshots(), writer, memb);
    // auto end = NOW;
    // std::cout << "duration of member " << memb << ": " << DURATION(end -
    // start) << " ms" << std::endl;
  }
}

template <typename Scalar>
void run_simulation(const NetCDFPDEReader &reader,
                    zisa::device_type memory_location) {
  int n_coupled = reader.get_n_coupled();
  int coupled_order = reader.get_coupled_function_order();

  zisa::array<Scalar, 1> function_scalings(
      zisa::shape_t<1>(n_coupled *
                       (unsigned int)std::pow(coupled_order, n_coupled)),
      memory_location);

  CoupledFunction<Scalar> func_coupled(function_scalings.const_view(),
                                       n_coupled, coupled_order);
  BoundaryCondition bc;
  int boundary_value = reader.get_boundary_value();
  if (boundary_value == 0) {
    bc = BoundaryCondition::Dirichlet;
  } else if (boundary_value == 1) {
    bc = BoundaryCondition::Neumann;
  } else if (boundary_value == 2) {
    bc = BoundaryCondition::Periodic;
  } else if (boundary_value == 3) {
    bc = BoundaryCondition::SpecialSG;
  } else {
    std::cout << "boundary condition not in range! " << std::endl;
    exit(-1);
  }

  // 0->Heat, 1->Wave
  int pde_type = reader.get_equation_type();
  if (pde_type == 0) {
    switch (reader.get_n_coupled()) {
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 1, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 2, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 3, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 4, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEHeat, 5, memory_location)
    default:
      std::cout << "only implemented for n_coupled <= 5 yet" << std::endl;
      exit(-1);
    }
  } else if (pde_type == 1) {
    std::cout << "GOT " << reader.get_n_coupled() << "\n";
    switch (reader.get_n_coupled()) {
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 1, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 2, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 3, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 4, memory_location)
      INSTANCIATE_PDE_AND_CALCULATE(PDEWave, 5, memory_location)
    default:
      std::cout << "only implemented for n_coupled <= 5 yet" << std::endl;
      exit(-1);
    }
  } else {
    std::cout << "pde type " << pde_type << " not implemented yet! "
              << std::endl;
    exit(-1);
  }
}

#endif // RUN_FROM_NETCDF_HPP
