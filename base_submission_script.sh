#!/bin/bash
#SBATCH -p rise # partition (queue)
#SBATCH -N 1 # number of nodes requested
#SBATCH -n 1 # number of tasks (i.e. processes)
#SBATCH --cpus-per-task=6 # number of cores per task
#SBATCH --gres=gpu:1
#SBATCH --nodelist=freddie
#SBATCH -t 0-12:00 # time requested (D-HH:MM)
#SBATCH -D /home/eecs/neilmrp # slurm will cd to this directory before running the script

echo starting job...
source ~/.bashrc
conda activate nfm_mods
cd ~/NFM_mod
# python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 --mult_noise_level 0.2 --manifold_mixup 1 --seed 1
# python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 --mult_noise_level 0.2 --manifold_mixup 1 --seed 1 --add_trigger patch
python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 --mult_noise_level 0.2 --manifold_mixup 1 --seed 1
python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 - 0.2 --manifold_mixup 1 --seed 1 --add_trigger patch --trigger_severity 0.05
python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 - 0.2 --manifold_mixup 1 --seed 1 --add_trigger patch --trigger_severity 0.1
python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 - 0.2 --manifold_mixup 1 --seed 1 --add_trigger patch --trigger_severity 0.15
python train_cifar.py --arch preactresnet18 --alpha 1.0 --add_noise_level 0.4 - 0.2 --manifold_mixup 1 --seed 1 --add_trigger patch --trigger_severity 0.20
