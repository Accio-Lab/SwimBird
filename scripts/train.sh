#!/bin/bash

export WANDB_DISABLED=true
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

NPROC_PER_NODE=8
WORLD_SIZE=1
RANK=0
MASTER_ADDR=0.0.0.0
MASTER_PORT=29502

DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# model configs
MODEL_SIZE='8B'
MODEL_NAME="Qwen/Qwen3-VL-${MODEL_SIZE}-Instruct"


DATA_PATH=(
    "SwimBird-SFT-92K/SwimBird-ZebraCoT"
    "SwimBird-SFT-92K/SwimBird-ThinkMorph"
    "SwimBird-SFT-92K/SwimBird-MathCanvas"
    "SwimBird-SFT-92K/SwimBird-OpenMMReasoner"
)

RANDOM_SEED=42
GRAD_CHECK=True

GLOBAL_BATCH_SIZE=128      
BATCH_PER_DEVICE=1
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NPROC_PER_NODE * WORLD_SIZE)))

# LLM-related params
LR=1e-5

# Latent-related params
LATENT_LOSS=mse
LATENT_LAMBDA=0.2
MAX_LATENT_TOKEN=32

MAX_TOKEN=16384
MIN_TOKEN=2

RUN_NAME="swimbird"
OUTPUT_DIR="swimbird"

export PYTHONPATH=$(pwd)
torchrun $DISTRIBUTED_ARGS \
    src/train/train.py \
    --run_name "$RUN_NAME" \
    --deepspeed scripts/zero2.json \
    --latent_loss $LATENT_LOSS\
    --model_id $MODEL_NAME \
    --data_path "${DATA_PATH[@]}" \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_merger True \
    --freeze_llm False \
    --learning_rate $LR \
    --latent_lambda $LATENT_LAMBDA \
    --max_latent_token $MAX_LATENT_TOKEN \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((MIN_TOKEN * 32 * 32)) \
    --image_max_pixels $((MAX_TOKEN * 32 * 32)) \
    --weight_decay 0.1 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 False \
    --gradient_checkpointing $GRAD_CHECK \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 8 \
    --random_seed $RANDOM_SEED \
    --report_to tensorboard # wandb