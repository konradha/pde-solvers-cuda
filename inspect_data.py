import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np

from sys import argv

def analytical_soliton_solution(X, Y, t):
    return 4 * np.arctan(np.exp(X + Y - t))


def plot_netcdf_surface(file_path, variable='data', n_coupled_index=0):
    with nc.Dataset(file_path, 'r') as f: 
        #print(f)
        data = f.variables[variable][:]

        xL, yL = f.x_length, f.y_length
        nx, ny = f.n_x, f.n_y
        xn = np.linspace(0, xL, nx)
        yn = np.linspace(0, yL, ny)
        T = f.final_time
        # TODO need number of timesteps as well in output, not only 
        # number of snapshots
        X, Y = np.meshgrid(xn, yn)
        n_snapshots = data.shape[1]

        dt = T / n_snapshots

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, data[0, 0,][1:-1, 1:-1], cmap='viridis',)
        plt.show()
        

        if n_snapshots <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            coupled_idx = n_coupled_index

            for i in range(n_snapshots):
                plot_data = data[0, i, :, coupled_idx::2] if n_coupled_idx > 1 else data[0, i]
                im = axes[i].imshow(plot_data, 
                                    cmap='viridis', 
                                    origin='lower', 
                                    extent=[0, 1, 0, 1]) 
                axes[i].set_title(f'snapshot {i+1}')
                axes[i].set_xlabel('x')
                axes[i].set_ylabel('y')
                
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            plt.tight_layout()
            fig.suptitle('NetCDF Data Surface Plots', fontsize=16)
            plt.subplots_adjust(top=0.93)
        else:
            from matplotlib.animation import FuncAnimation
            coupled_idx = n_coupled_index 
            fig = plt.figure(figsize=(20, 20))
            ax = fig.add_subplot(111, projection='3d')
            if coupled_idx > 0:
                surf = ax.plot_surface(X, Y, data[0, 0, :, coupled_idx::2][1:-1, 1:-1], cmap='viridis',)
            else:
                surf = ax.plot_surface(X, Y, data[0, 0,][1:-1, 1:-1], cmap='viridis',)
             
            def update(frame):
                ax.clear()
                if coupled_idx > 0:
                    ax.plot_surface(X, Y, (data[0, frame, :, coupled_idx::2][1:-1, 1:-1]), cmap='viridis')
                else:
                    #print("plotting this guy")
                    ax.plot_surface(X, Y,
                            np.abs(data[0, frame,][1:-1, 1:-1] - analytical_soliton_solution(X, Y, frame * dt)),
                            #(data[0, frame,][1:-1, 1:-1]),
                            cmap='viridis')

            fps = 300
            ani = FuncAnimation(fig, update, frames=n_snapshots, interval=n_snapshots / fps, )
            plt.show()

    return fig

if __name__ == '__main__':
    fname = str(argv[1]) 
    fig = plot_netcdf_surface(fname)
    plt.show()
