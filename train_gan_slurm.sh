#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name spectral-vd-euxfel
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate spectral-vd-euxfel
cd /home/kaiserja/beegfs/spectral-vd-euxfel

srun python train_gan.py
# wandb agent --count 1 msk-ipc/ares-ea-v2/3z55mih5

exit
