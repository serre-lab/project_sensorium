#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 1
#SBATCH -N 1 
#SBATCH --mem=80GB
#SBATCH -J color_CNN
#SBATCH -C quadrortx
##SBATCH --constraint=v100
#SBATCH -o /cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/logs/arjun_MI_%A_%a_%J.out
#SBATCH -e /cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/logs/arjun_MI_%A_%a_%J.err
##SBATCH --account=carney-tserre-condo
##SBATCH --array=0-1

##SBATCH -p gpu

cd /cifs/data/tserre/CLPS_Serre_Lab/projects/prj_sensorium/arjun/
echo "Switched to project directory."

module load anaconda/3-5.2.0
module load python/3.5.2

# if [[ $USER -eq "anagara8" ]]
# then
#     	source activate pytorch
# else
#     	source activate color_CNN
# fi

source activate color_CNN

echo $SLURM_ARRAY_TASK_ID

# wandb login
# wandb init

python -u trainer.py $USER
