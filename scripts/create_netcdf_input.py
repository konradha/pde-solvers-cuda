'''
This file should be an easy template to generate netcdf input files for the pde-solvers-cuda function.
You can also create this files by yourself with another method but the file has to have following structure:

NetCDFFile {
    Attributes {
        type_of_equation-> int (0=HeatEquation, 1=WaveEquation)
        n_members -> int (number of different problems with own initial condition, sigma values and function scalings)

        x_size -> int (number of nodes OF ONE PDE in x-direction WITHOUT boundary condition nodes)
        x_length -> Scalar (length of grid in x-direction)
        y_size -> int (number of nodes OF ONE PDE in y-direction WITHOUT boundary condition nodes)
        y_length -> Scalar (length of grid in y-direction)

        boundary_value_type -> int (0=Dirichlet, 1=Neumann, 2=Periodic)
        scalar_type -> int (0=float, 1=double)

        n_coupled -> int (#number coupled PDEs)
        coupled_function_order -> int (order of coupled function, 
                                     example: n_coupled=2, coupled_function_order=2 => f(x, y) = a + bx + cy + dxy)
        number_timesteps -> int (number of timesteps of simulation)
        final_time -> Scalar (final time after simulation)
        number_snapshots -> int (number snapshots to shoot, includes initial time values)

        file_to_save_output -> string (file to save output)
    }

    Variables {
        initial_data -> scalar_type^(n_members, x_size + 2, n_coupled * (y_size + 2))
        sigma_values -> scalar_type^(n_members, 2*x_size - 3, y_size - 1)
        [bc_neumann_values -> scalar_type^(n_members, x_size + 2, n_coupled * (y_size + 2))] (only if Boundary_value_type=1)
        function_scalings -> scalar_type^(n_members, n_coupled * coupled_function_order)
    }
}

The data of sigma values is arranged in following way:


   x_00              x_01              x_02              x_03
                       |                 |
                     sigma_00          sigma_01
                       |                 |
   x_10 - sigma_10 - x_11 - sigma_11 - x_12 - sigma_12 - x_13
                       |                 |
                     sigma_20          sigma_21
                       |                 |
   x_20              x_21              x_22              x_23

                                ||
                                \/

                sigma_00, sigma_01, 0
                sigma_10, sigma_11, sigma_12
                sigma_20, sigma_21, 0

'''
import netCDF4 as nc
import numpy as np
    
# function template for initial_values, bc_values and sigma_values
# note that here the arguments are the member and the x and y-positions in the grid
def dummy_function_2d(member: int, coupled_idx: int, x_position: float, y_position: float):
    if member == 0:
        return x_position * y_position + y_position
    else:
        return 0.001

# function template for function_scalings
# note that here the arguments are the member total amount of funciton variables (n_coupled * max_order)
def dummy_function_scalings(member, size: int):
    x_values = np.linspace(0, size-1, num=size)
    return member * x_values + 2

    
# make shure that you input the right types, for example in final_time you have to input a float (1. and not 1)
def create_input_file(filename, file_to_save_output, type_of_equation=0, 
                      x_size=8, x_length=1., y_size=8, y_length=1., boundary_value_type=1,
                      scalar_type=0, n_coupled=1, 
                      coupled_function_order=2, number_timesteps=1000,
                      final_time=1., number_snapshots=3, n_members=2, initial_value_function=dummy_function_2d,
                      sigma_function=dummy_function_2d, bc_neumann_function=dummy_function_2d, f_value_function=dummy_function_scalings):

    # Create a new NetCDF file
    with nc.Dataset(filename, 'w') as root:
        # Define attributes
        root.type_of_equation = type_of_equation  # 0=HeatEquation, 1=WaveEquation
        root.n_members = n_members # number of different initial_conditions, boundary_values and sigma_values

        root.x_size = x_size  # Number of nodes OF ONE PDE in x-direction WITHOUT boundary condition nodes
        root.x_length = x_length # length of grid in x-direction

        root.y_size = y_size  # Number of nodes OF ONE PDE in y-direction WITHOUT boundary condition nodes
        root.y_length = y_length # length of grid in y-direction

        root.boundary_value_type = boundary_value_type  # 0=Dirichlet, 1=Neumann, 2=Periodic
        root.scalar_type = scalar_type  # 0=float, 1=double

        root.n_coupled = n_coupled  # Number of coupled PDEs
        root.coupled_function_order = coupled_function_order  # Order of coupled function

        root.number_timesteps = number_timesteps # number of timesteps of simulation
        root.final_time = final_time # final time after simulation
        root.number_snapshots = number_snapshots # number snapshots to shoot, includes initial time values

        root.file_to_save_output = file_to_save_output


        # Define variables
        if scalar_type == 0:
            scalar_type_string = 'f4' # float
        elif scalar_type == 1:
            scalar_type_string = 'f8' # double
        else:
            print("error: only scalar_type 0 or 1 allowed!")
            exit(-1)
        x_size_sigma = 2 * x_size + 1
        y_size_sigma = y_size + 1

        x_size_dim = root.createDimension("x_size", x_size + 2)
        y_size_dim = root.createDimension("y_size", n_coupled * (y_size + 2))
        x_size_sigma_dim = root.createDimension("x_size_sigma", x_size_sigma)
        y_size_sigma_dim = root.createDimension("y_size_sigma", y_size_sigma)
        n_members_dim = root.createDimension("n_members", n_members)
        function_scalings_dim = root.createDimension("function_scaling_size", n_coupled * coupled_function_order)

        # store in right chunksize to enable fast loading of variables
        initial_data = root.createVariable('initial_data', scalar_type_string, dimensions=('n_members', 'x_size', 'y_size'), chunksizes=(1, x_size, y_size))
        sigma_values = root.createVariable('sigma_values', scalar_type_string, dimensions=('n_members', 'x_size_sigma', 'y_size_sigma'), chunksizes=(1, x_size_sigma, y_size_sigma))
        function_scalings = root.createVariable('function_scalings', scalar_type_string, dimensions=('n_members', 'function_scaling_size'), chunksizes=(1, n_coupled * coupled_function_order))
        if root.boundary_value_type == 1 or root.type_of_equation == 1:
            bc_neumann_values = root.createVariable('bc_neumann_values', scalar_type_string, ('n_members', 'x_size', 'y_size'))

        function_scalings[:, :] = np.linspace(0., 1., n_coupled * coupled_function_order)

        # add values on boundary
        # if you have dirichlet or neumann bc the initial_value_function should evaluate on the boundary too
        # if you have periodic bc the boundary values will get adjusted later in the algorithm
        dx = x_length / (x_size - 1)
        dy = y_length / (y_size - 1)
        x_positions = np.linspace(- dx, x_length + dx, x_size + 2)
        y_positions = np.linspace(- dy, y_length + dy, y_size + 2)
        xx, yy = np.meshgrid(x_positions, y_positions, indexing='ij')


        
        x_positions_sigma = np.linspace(- dx * 0.5, x_length + dx * 0.5, 2 * x_size + 1)
        y_positions_sigma = np.linspace(- dy * 0.5, y_length + dy * 0.5, y_size + 1)
        xx_sigma, yy_sigma = np.meshgrid(x_positions_sigma, y_positions_sigma, indexing='ij')
        # adjust y-values of every second row because those are shifted
        yy_sigma[::2, :] += dy * .5

        for member in range(n_members):
            function_scalings[member, :] = f_value_function(member, n_coupled * coupled_function_order)
            for coupled_idx in range(n_coupled):
                initial_data[member, :, coupled_idx::n_coupled] = initial_value_function(member, coupled_idx, xx, yy)
                sigma_values[member, :, coupled_idx::n_coupled] = sigma_function(member, coupled_idx, xx_sigma, yy_sigma)
                # If boundary_value_type is Neumann, define additional variable
                if root.boundary_value_type == 1 or root.type_of_equation == 1:
                    bc_neumann_values[member, :, coupled_idx::n_coupled] = bc_neumann_function(member, xx, yy)
        # print(initial_data[0, :, :])


    print(f"NetCDF file '{filename}' created successfully.")



def bc_neumann_function(member: int, coupled_idx: int, x: float, y: float):
    if member == 0:
        return x * 0.04 + y * 0.01
    if member == 1:
        return 0.

def function_scalings_zero(member: int, size: int):
    x_values = np.linspace(0, size-1, num=size)
    return 0 * x_values


# dot in the middle
def dot_function_2d(member: int, coupled_idx: int, x_position: float, y_position: float):
    if member == 0:
        return np.exp(- (x_position-0.5)**2 - (y_position-0.5)**2)
    else:
        return x_position * y_position  + y_position

def linear_damping(member: int, size: int):
    f_values = np.zeros(size)
    if member == 0:
        f_values[1] = -5
    if member == 1:
        f_values[0] = 10
    return f_values

def zero_f(member: int, size: int):
    return np.zeros(size)

def const_function_2d(member: int, coupled_idx: int, x_position: float, y_position: float):
    return 0.1

def hat_functions_2d(member: int, coupled_idx: int, x_position: float, y_position: float):
    u = 0. * x_position
    nx = x_position.shape[0]
    ny = x_position.shape[1]

    u[ nx//4:nx//4+10 , ny//4:ny//4+10 ] = 2
    u[ 3*nx//4:3*nx//4+10 , ny//4:ny//4+10 ] = 3
    u[ nx//4:nx//4+10 , 3*ny//4:3*ny//4+10 ] = 4
    u[ 3*ny//4:3*ny//4+10 , 3*ny//4:3*ny//4+10 ] = 5    

    return u

if __name__ == "__main__":
    # Usage example:
    create_input_file('data/example.nc', 'data/example_out.nc', type_of_equation=0, 
                      x_size=160, x_length=2., y_size=160, y_length=2., boundary_value_type=0,
                      scalar_type=0, n_coupled=1, 
                      coupled_function_order=1, number_timesteps=5000,
                      final_time=0.2, number_snapshots=6, n_members=1, initial_value_function=hat_functions_2d,
                      sigma_function=const_function_2d, bc_neumann_function=bc_neumann_function, f_value_function=function_scalings_zero)
    