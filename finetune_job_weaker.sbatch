#!/usr/bin/env bash
#SBATCH --job-name=internvl2_dpo
#SBATCH --partition=nvidia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=71:00:00
#SBATCH --constraint=80g
#SBATCH --output=/scratch/jdg8833/Code/InternVL/logs/%x-%j.out
#SBATCH --error=/scratch/jdg8833/Code/InternVL/logs/%x-%j.err


echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

sleep $(( (RANDOM%10) + 1 ))
echo "SLURM_JOBID: " $SLURM_JOBID
# --- Modules ---
module load cuda/12.2.0


# activate conda env
source ~/.bashrc
eval "$(conda shell.bash hook)"

conda activate internvl

cd /path/to/repo/InternVL/internvl_chat/
export PYTHONPATH=$PYTHONPATH:path/to/repo/InternVL/internvl_chat/
srun bash shell/internvl2.0_mpo/preference_optimization/internvl2_1b_dpo.sh
