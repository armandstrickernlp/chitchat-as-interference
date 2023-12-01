#!/bin/bash
#SBATCH --job-name=S_S
#SBATCH --output=local_logs/s_s.out 
#SBATCH --error=local_logs/s_s.err 
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=15:00:00
#SBATCH --hint=nomultithread 
#SBATCH --account=rqz@a100
#SBATCH --array=0-4

args=()

for path in training_outputs/simpletod/3e-05_42_rank64/checkpoint-1400 training_outputs/simpletod/3e-05_43_rank64/checkpoint-1400 training_outputs/simpletod/3e-05_44_rank64/checkpoint-1400 training_outputs/simpletod/3e-05_45_rank64/checkpoint-1400 training_outputs/simpletod/3e-05_46_rank64/checkpoint-1400
do 
    args+=("--checkpoint_path ${path}")
done

srun python gen_and_evaluate.py --training_set=simpletod --eval_data_json=data/lm_data/simpletod.json --eval_split=test ${args[${SLURM_ARRAY_TASK_ID}]}


