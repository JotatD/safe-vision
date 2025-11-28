set -x

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-256}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TRITON_CACHE_DIR="/tmp/triton_wwy/"
export MASTER_PORT=34229

OUTPUT_DIR='work_dirs/internvl_chat_v2_dpo/Internvl2-1B_1000'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# NOTE: you can download MMPR-v1.2 from: https://huggingface.co/datasets/OpenGVLab/MMPR-v1.2

python -u internvl/train/internvl_chat_dpo.py \
  --model_name_or_path "OpenGVLab/InternVL2-1B" \
  --conv_style "internvl2_5" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path ""/scratch/jdg8833/Dataset/internvl_large_1000/meta.json"" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 200 \
  --save_total_limit 100 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 6144 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "wandb" \
  --loss_type sigmoid,bco_pair \
  --sigmoid_loss_weight 0.8 \
  --bco_pair_loss_weight 0.2 \
  --rpo_alpha 1 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
