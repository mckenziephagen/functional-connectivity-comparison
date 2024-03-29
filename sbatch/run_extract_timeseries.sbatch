#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=901
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m3513


#SBATCH --nodes=1
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=2

source ${HOME}/.bashrc

conda activate fc_w_datalad


for i in {0..100}; do
    subject_id=$( sed -n $((SLURM_ARRAY_TASK_ID+$i))p ${PSCRATCH}/hcp-functional-connectivity/subjects.txt )
    echo $subject_id
    srun --ntasks=1 --mem=8G --exclusive python extract_timeseries.py --subject_id $subject_id &

done
wait

