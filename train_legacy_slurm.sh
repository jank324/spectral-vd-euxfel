#!/bin/sh
#SBATCH --partition=maxgpu
#SBATCH --job-name ml-lps-euxfel
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --constraint=P100|V100
#SBATCH --mail-type ALL

source /etc/profile.d/modules.sh

source ~/.bashrc
conda activate spectral-vd-euxfel
cd /home/kaiserja/beegfs/spectral-vd-euxfel

# srun python train_legacy_current.py
# srun wandb agent --count 1 msk-ipc/virtual-diagnostics-euxfel-current-legacy/890897om

# srun python train_legacy_lps.py
srun wandb agent --count 1 msk-ipc/virtual-diagnostics-euxfel-lps-legacy/oh87rxdm

exit
