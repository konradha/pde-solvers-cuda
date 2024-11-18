# pde-solvers-cuda


## Requirements
Currently, only UNIX based operating systems are supported. Moreover, you need to have the following installed on your machine:
* C++ compiler (e.g. gcc, clang)
* CMake (at least VERSION 3.18)
* CUDA (not strictly needed)
* git (not strictly needed, you could also download the repo as .zip file)
* hdf5
* netcdf
* python

## Getting started
You can build pds-solvers-cuda as follows

```
git clone https://github.com/LouisHurschler/pde-solvers-cuda.git  
cd pde-solvers-cuda
mkdir build
cmake -B build -DENABLE_CUDA={ON, OFF}
```
Note that `ENABLE_CUDA` is set `OFF` by default

## Running the application
The easiest way to run the application is by following these steps:

### Create a NetCDF Input File
Use the Python function in `scripts/create_netcdf_input.py` to create a NetCDF input file. 
You can refer to the example provided in `scripts/brusselator_script.py`.

### Run the Solver
After you have generated the input file, run the solver using:
```
./build/run_from_netcdf <filename> [1]
```

where `<filename>` denotes the path of the file generated by your 
script relative to the directory you're running the application.

This will run on the GPU if you have a GPU available, you have
compiled the application with `-DENABLE_CUDA=ON` and if you 
run it with the optional `1` added at the end.

### Output
After the executable finishes, the results are stored in
a NetCDF file stored at the location specified in your input script.

You can use the Python function in `scripts/plot_output_file.py` to plot the output or use the NetCDF
output otherwise.

