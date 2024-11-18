#!/bin/bash
#SBATCH --job-name=make_pde_solver
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=make_job_%j.out

cd $(pwd)/build
make -j8
