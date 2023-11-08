#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=1-100 #1-100%150
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m3513


#SBATCH --nodes=1
#SBATCH --cpus-per-task=2

#activate conda envs
source ${HOME}/.bashrc
conda activate fc

#parse subject ID from text file
subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID))p test_subjects.txt )
echo $subject_id

srun python calc_fc.py --subject_id $subject_id --model 'uoi-lasso'

wait