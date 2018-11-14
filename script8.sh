#!/bin/sh 
#SBATCH -o gpu-job-%j.output 
#SBATCH -p K20q 
#SBATCH --gres=gpu:2
#SBATCH -n 1 
 
module load cuda90/toolkit  cuda90/blas/9.0.176 

/usr/bin/python3.5 starter.py
