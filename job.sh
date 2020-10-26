#!/bin/bash                                                                 
#SBATCH --job-name=cluster   # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --ntasks=1                # Number of MPI ranks
#SBATCH --cpus-per-task=1           # Number of cores per MPI rank 
#SBATCH --gpus-per-task=1
#SBATCH --reservation=perez
#SBATCH --partition=gpu
##SBATCH --nodes=1                  # Number of nodes
#SBATCH --mem-per-cpu=100gb          # Memory per processor
#SBATCH --time=60:00:00              # Time limit hrs:min:sec
#SBATCH --output=hmm.log     # Standard output and error log
##SBATCH --mail-user=secondliwei@gmail.com
##SBATCH  --qos=alberto.perezant-b
pwd; hostname; date
                   
source ~/.bashrc 
conda activate torch      
which python
export PYTHONPATH=/home/liweichang/anaconda3/envs/torch/lib/python3.6/site-packages/:$PYTHONPATH
jupyter notebook --ip $(hostname) --no-browser
