#!/bin/bash
#SBATCH --partition=gpgpuD,gpgpuC, gpgpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ik323
export PATH=/vol/bitbucket/ik323/myvenv/bin/:$PATH
source activate
source /vol/cuda/12.2.0/setup.sh
python trainer.py
uptime
