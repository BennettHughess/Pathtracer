#!/bin/bash
#SBATCH --account=PAS2137
#SBATCH --job-name blackhole_test
#SBATCH --nodes=1
#SBATCH --time=00:00:30
#SBATCH --gpus-per-node=1

module load cuda

date
/users/PAS2137/bennetthughes137/Pathtracer/bin/main
date

#run the job with "sbatch job.sh"