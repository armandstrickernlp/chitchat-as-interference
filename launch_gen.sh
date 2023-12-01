#!/bin/bash
#SBATCH --job-name=TRAIN0
#SBATCH --output=local_logs/gen_interfer_train_0.out 
#SBATCH --error=local_logs/gen_interfer_train_0.err 
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=14
#SBATCH --time=8:00:00
#SBATCH --hint=nomultithread 
#SBATCH --account=rqz@a100

srun python gen_chitchat.py --gen_sit_path=outputs/VALID/gen_situations.json