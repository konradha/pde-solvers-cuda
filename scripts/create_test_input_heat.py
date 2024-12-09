from create_netcdf_input import *

# test case from chapter 3.5 in thesis
def initial_function(member, coupled_idx, x_position, y_position):
    if coupled_idx == 0:
        #u = np.exp(-((x_position -.5) ** 2 + (y_position - .5) ** 2))
        u = np.exp((x_position + y_position ))
    elif coupled_idx == 1:
        # initial velocity 0?
        u = np.zeros(shape=x_position.shape)
    else:
        print("initial function is only meant for n_coupled == 2!")
        u = np.zeros(shape=x_position.shape)
    return u  


def lin_func(member, coupled_idx, x_position, y_position):
    return np.ones(shape=x_position.shape) * 1000. 

def sigma(member, x_position, y_position):
    return np.exp(-x_position - y_position) +  0.25

def f(member, size):
    f = np.zeros(size)
    f[0] = 1000.
    f[2] = -0.5
    return f


create_input_file('data/test_heat.nc', 'data/test_heat_out.nc', type_of_equation=0, 
                      x_size=128, x_length=1., y_size=128, y_length=1., boundary_value_type=1,
                      scalar_type=0, n_coupled=2, 
                      coupled_function_order=3, number_timesteps=5000,
                      final_time=.01, number_snapshots=100, n_members=1, initial_value_function=initial_function,
                      sigma_function=sigma, bc_neumann_function=lin_func, f_value_function=f)
