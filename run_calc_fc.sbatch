#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=15:00
#SBATCH --constraint=cpu
#SBATCH --array=1 #1-100%150
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m3513

#SBATCH --nodes=1
#SBATCH --ntasks=100

#activate conda envs
source ${HOME}/.bashrc
conda activate fc

for i in {0..100}; do
#parse subject ID from text file
    subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID+$i))p test_subjects.txt )
    echo $subject_id

    srun -n 1 -c 2 python calc_fc.py --subject_id $subject_id --model 'correlation' &
    
done

wait



