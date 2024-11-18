#!/bin/bash
#SBATCH --qos=debug
#SBATCH --time=30:00
#SBATCH --constraint=cpu
#SBATCH --array=1
#SBATCH --error logs/%A_%a.error
#SBATCH --out logs/%A_%a.out
#SBATCH --account=m3513
#SBATCH --mail-user=mphagen@uw.edu
#SBATCH --mail-type=END


#SBATCH --nodes=1
#SBATCH --ntasks=200
#SBATCH --cpus-per-task=1

source ${HOME}/.bashrc

script_path="$HOME/functional_connectivity_comparison/processing/extract_timeseries.py" 

conda activate fc_w_datalad


for i in {0..100}; do
    subject_id=$( sed -n $(($SLURM_ARRAY_TASK_ID+$i))p ${PSCRATCH}/hcp-functional-connectivity/subjects.txt )
    echo $subject_id
    srun --ntasks=1 python $script_path --subject_id $subject_id &

done
wait

