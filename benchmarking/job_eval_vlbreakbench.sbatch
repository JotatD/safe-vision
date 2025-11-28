#!/bin/bash
#SBATCH --job-name=vlbreakbench_eval
#SBATCH --output=/l/users/mashrafi.monon/vlm_jailbreak_attack_defense/benchmarking/logs/vlbreakbench_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

source ~/.bashc
conda activate openflamingo
module load cuda-11.6

cd /l/users/mashrafi.monon/vlm_jailbreak_attack_defense

python benchmarking/eval_openflamingo_vljailbreak.py \
  --dataset-root datasets/VLBreakBench \
  --annotations vlbreakbench_base.json \
  --hf-repo-id openflamingo/OpenFlamingo-9B-vitl-mpt7b \
  --checkpoint-filename checkpoint.pt \
  --output-dir benchmarking/results/vlbreakbench_openflamingo \
  --decoder-layers-attr-name transformer.blocks \
  --trust-remote-code \
  --skip-gemini
