#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=30:00
#SBATCH --constraint=cpu

#SBATCH --array=33 #1-100%150
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out

#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --account=m1266

source ${HOME}/.bashrc

conda activate fc_w_datalad

for i in {0..9}; do

    subject_id=$( sed -n $((SLURM_ARRAY_TASK_ID + i))p test_subjects.txt )
    echo $subject_id
    srun -n-tasks 1 --exclusive python extract_timeseries.py --subject_id $subject_id &

done

wait

