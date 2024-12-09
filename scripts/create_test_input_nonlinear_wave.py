from create_netcdf_input import *

def initial_function(member, coupled_idx, x_position, y_position):
    if member == 0 and coupled_idx == 0:
        return 4 * np.arctan(np.exp(x_position + y_position))

def init_deriv(member, coupled_idx, x_position, y_position):
    if member == 0 and coupled_idx == 0:
        return -4 * np.exp(x_position + y_position) / (1. + np.exp(2 * x_position + 2 * y_position))
        

def sigma(member, x_position, y_position):
    if member == 0:
        return np.ones(shape=x_position.shape)

def scaling_function(member, size):
    return np.ones(size)

create_input_file('data/test_nonlinear_wave.nc', 'data/test_nonlinear_wave_out.nc',
                    type_of_equation=1, 
                    x_size=64, x_length=5.,
                    y_size=64, y_length=5.,
                    boundary_value_type=3, # u_x = f(y, t) and u_y = g(x, t) for all t \in [0 + dt, T)
                    scalar_type=0, n_coupled=1, 
                    coupled_function_order=3, number_timesteps=5000,
                    final_time=10.,
                    number_snapshots=5000,
                    n_members=1,
                    initial_value_function=initial_function,
                    sigma_function=sigma,
                    bc_neumann_function=init_deriv,
                    f_value_function=scaling_function,
                    extra_source_term=0, # -sin(u)
                    extra_source_eps=1., # ie u_tt = \Delta u - eps * F(u) = \Delta u - sin(u) 
                    )
