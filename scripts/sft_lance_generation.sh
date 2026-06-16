#!/bin/bash

# ==============================Manual launch mode==============================
MANUAL_START=${MANUAL_START:-True}  # True: manual launch; False: automatic launch

if [ "$MANUAL_START" = "True" ]; then
    # for one machine
    export ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-1} # number of machines
    if [ -z "${ARNOLD_WORKER_GPU:-}" ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            ARNOLD_WORKER_GPU=$(nvidia-smi -L | wc -l)
        else
            ARNOLD_WORKER_GPU=1
        fi
        export ARNOLD_WORKER_GPU
    fi
    export ARNOLD_ID=${ARNOLD_ID:-0} # machine ID
    MAIN_PROCESS_IP=127.0.0.1
    MAIN_PROCESS_PORT=${ARNOLD_WORKER_0_PORT:-6668}
    MACHINE_RANK=$ARNOLD_ID

    # # for multiple machines
    # export ARNOLD_WORKER_NUM=1 # 2 # number of machines
    # OFFSET=0
    # eval "MAIN_PROCESS_IP=\$ARNOLD_WORKER_${OFFSET}_HOST"
    # eval "MAIN_PROCESS_PORT=\$(echo '$ARNOLD_WORKER_0_PORT' | cut -d "," -f 1)"
    # MACHINE_RANK=$((ARNOLD_ID - OFFSET))
else
    MACHINE_RANK=$ARNOLD_ID
fi

# ==============================Distributed configuration==============================
export NCCL_DEBUG=WARN  # Log warnings and errors only
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore::FutureWarning:deepspeed.runtime.zero.linear"
TOTAL_RANK=$((ARNOLD_WORKER_NUM * ARNOLD_WORKER_GPU))
NUM_SHARD=${NUM_SHARD:-$((TOTAL_RANK < 4 ? TOTAL_RANK : 4))}
if [ "$TOTAL_RANK" -lt 1 ]; then
    echo "TOTAL_RANK must be >= 1, got $TOTAL_RANK"
    exit 1
fi
if [ "$NUM_SHARD" -lt 1 ] || [ $((TOTAL_RANK % NUM_SHARD)) -ne 0 ]; then
    echo "NUM_SHARD must divide TOTAL_RANK. Got TOTAL_RANK=$TOTAL_RANK, NUM_SHARD=$NUM_SHARD"
    exit 1
fi
NUM_REPLICATE=$((TOTAL_RANK / NUM_SHARD))

echo "ARNOLD_WORKER_NUM: $ARNOLD_WORKER_NUM"
echo "ARNOLD_WORKER_GPU: $ARNOLD_WORKER_GPU"
echo "TOTAL_RANK: $TOTAL_RANK"
echo "MACHINE_RANK: $MACHINE_RANK"
echo "MAIN_PROCESS_IP: $MAIN_PROCESS_IP"
echo "MAIN_PROCESS_PORT: $MAIN_PROCESS_PORT"
echo "NUM_REPLICATE: $NUM_REPLICATE"
echo "NUM_SHARD: $NUM_SHARD"

# ==============================Model and initialization configuration==============================
VIT_PATH=./downloads/Qwen2.5-VL-ViT
MODEL_PATH=./downloads/Lance_3B_Video # or use ./downloads/Lance_3B for image-related task
VIT_TYPE="qwen_2_5_vl_original"

LOAD_FROM_LANCE_CHECKPOINT=True
INIT_FROM_VLM_CHECKPOINT=False
COPY_INIT_MOE=True

LLM_QK_NORM=True # Use QK_NORM in LLM_UND and LLM_GEN, only when both LLM_QK_NORM_UND and LLM_QK_NORM_GEN are False
LLM_QK_NORM_UND=True # Use QK_NORM in LLM_UND
LLM_QK_NORM_GEN=True # Use QK_NORM in LLM_GEN
TIE_WORD_EMBEDDINGS=False

LATENT_PATCH_SIZE="1 1 1"
MAX_LATENT_SIZE=64
MAX_NUM_FRAMES=121
VAE_MODEL_TYPE="wan"
VISUAL_UND=True
VISUAL_GEN=True # False # Set False for understanding-only tasks

# ==============================Freeze configuration==============================
FREEZE_LLM=False # Freeze all LLM_UND and LLM_GEN parameters
FREEZE_LLM_EMBED_TOKENS=True # Freeze the LLM embedding layer
FREEZE_UND_PARAMS=False # Freeze all LLM_UND parameters
FREEZE_UND=False # Freeze LLM_UND behavior

# ==============================Data configuration==============================
DATASET_CONFIG_FILE=${DATASET_CONFIG_FILE:-config/train_local/multi_gen.yaml}
VAL_DATASET_CONFIG_FILE=${VAL_DATASET_CONFIG_FILE:-config/train_local/multi_gen.yaml}

NUM_WORKERS=${NUM_WORKERS:-8}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
MAX_NUM_TOKENS=65536
EXPECTED_NUM_TOKENS=65536
MAX_NUM_TOKENS_PER_SAMPLE=65536
PREFER_BUFFER_BEFORE=65536
MAX_BUFFER_SIZE=50
REQUIRE_UND_GEN=True

DATA_SEED=2026
GLOBAL_SEED=2026

TEXT_COND_DROPOUT_PROB=0.1
VAE_COND_DROPOUT_PROB=0
VIT_COND_DROPOUT_PROB=0.1

# ==============================Optimizer and training configuration==============================
LR=0.0001
MIN_LR=0.000001
WARMUP_STEPS=500
LR_SCHEDULER="constant" # "cosine" | "constant"

CE_WEIGHT=0.25
MSE_WEIGHT=1.0
TIMESTEP_SHIFT=4.0
USE_FLEX=True

LOG_EVERY=2
SAVE_EVERY=10 # 200 # 1000
CKPT_DEBUG_STEPS=10

# ==============================EMA, resume, and output configuration==============================
USE_EMA=True
AUTO_RESUME=False
RESUME_MODEL_ONLY=True
LOAD_DATA_STATUS=False

DATE=v0608
WANDB_PROJECT=huangmengqi_sft_lance
WANDB_NAME=${DATE}_"$NUM_SHARD"x"$NUM_REPLICATE"_multi_gen
WANDB_OFFLINE=True # False

OUTPUTS_DIR=./outputs/${WANDB_PROJECT}

# ==============================Validation configuration==============================
VALIDATION_STEP=-1 # 100 | -1 disables validation
VALIDATION_TYPE="und_gen"
VALIDATION_NUM_TIMESTEPS=10
VALIDATION_TIMESTEP_SHIFT=3.0
VALIDATION_MAX_SAMPLES=$((128 / TOTAL_RANK))
VALIDATION_LOG_TYPE="direct"
echo "VALIDATION_MAX_SAMPLES: $VALIDATION_MAX_SAMPLES"

# ==============================Runtime switches==============================
APPLY_QWEN_2_5_VL_POS_EMB=True
APPLY_CHAT_TEMPLATE=False

# ==============================Launch training==============================
accelerate launch \
    --num_machines                              $ARNOLD_WORKER_NUM \
    --num_processes                             $TOTAL_RANK \
    --machine_rank                              $MACHINE_RANK \
    --main_process_ip                           $MAIN_PROCESS_IP \
    --main_process_port                         $MAIN_PROCESS_PORT \
    --mixed_precision                           bf16 \
    train/unified_train.py \
    --model_path                                $MODEL_PATH \
    --vit_path                                  $VIT_PATH \
    --vit_type                                  $VIT_TYPE \
    --llm_qk_norm                               $LLM_QK_NORM \
    --llm_qk_norm_und                           $LLM_QK_NORM_UND \
    --llm_qk_norm_gen                           $LLM_QK_NORM_GEN \
    --tie_word_embeddings                       $TIE_WORD_EMBEDDINGS \
    --max_num_frames                            $MAX_NUM_FRAMES \
    --max_latent_size                           $MAX_LATENT_SIZE \
    --latent_patch_size                         $LATENT_PATCH_SIZE \
    --load_from_lance_checkpoint                $LOAD_FROM_LANCE_CHECKPOINT \
    --init_from_vlm_checkpoint                  $INIT_FROM_VLM_CHECKPOINT \
    --copy_init_moe                             $COPY_INIT_MOE \
    --visual_und                                $VISUAL_UND \
    --visual_gen                                $VISUAL_GEN \
    --vae_model_type                            $VAE_MODEL_TYPE \
    --freeze_llm                                $FREEZE_LLM \
    --freeze_llm_embed_tokens                   $FREEZE_LLM_EMBED_TOKENS \
    --freeze_und_params                         $FREEZE_UND_PARAMS \
    --freeze_und                                $FREEZE_UND \
    --dataset_config_file                       $DATASET_CONFIG_FILE \
    --val_dataset_config_file                   $VAL_DATASET_CONFIG_FILE \
    --num_workers                               $NUM_WORKERS \
    --prefetch_factor                           $PREFETCH_FACTOR \
    --expected_num_tokens                       $EXPECTED_NUM_TOKENS \
    --max_num_tokens                            $MAX_NUM_TOKENS \
    --max_num_tokens_per_sample                 $MAX_NUM_TOKENS_PER_SAMPLE \
    --prefer_buffer_before                      $PREFER_BUFFER_BEFORE \
    --max_buffer_size                           $MAX_BUFFER_SIZE \
    --require_und_gen                           $REQUIRE_UND_GEN \
    --data_seed                                 $DATA_SEED \
    --global_seed                               $GLOBAL_SEED \
    --text_cond_dropout_prob                    $TEXT_COND_DROPOUT_PROB \
    --vae_cond_dropout_prob                     $VAE_COND_DROPOUT_PROB \
    --vit_cond_dropout_prob                     $VIT_COND_DROPOUT_PROB \
    --warmup_steps                              $WARMUP_STEPS \
    --lr_scheduler                              $LR_SCHEDULER \
    --lr                                        $LR \
    --min_lr                                    $MIN_LR \
    --ce_weight                                 $CE_WEIGHT \
    --mse_weight                                $MSE_WEIGHT \
    --timestep_shift                            $TIMESTEP_SHIFT \
    --use_flex                                  $USE_FLEX \
    --num_replicate                             $NUM_REPLICATE \
    --num_shard                                 $NUM_SHARD \
    --use_ema                                   $USE_EMA \
    --auto_resume                               $AUTO_RESUME \
    --resume_model_only                         $RESUME_MODEL_ONLY \
    --load_data_status                          $LOAD_DATA_STATUS \
    --outputs_dir                               $OUTPUTS_DIR \
    --save_every                                $SAVE_EVERY \
    --log_every                                 $LOG_EVERY \
    --ckpt_debug_steps                          $CKPT_DEBUG_STEPS \
    --wandb_project                             $WANDB_PROJECT \
    --wandb_name                                $WANDB_NAME \
    --wandb_offline                             $WANDB_OFFLINE \
    --validation_step                           $VALIDATION_STEP \
    --validation_type                           $VALIDATION_TYPE \
    --validation_num_timesteps                  $VALIDATION_NUM_TIMESTEPS \
    --validation_timestep_shift                 $VALIDATION_TIMESTEP_SHIFT \
    --validation_max_samples                    $VALIDATION_MAX_SAMPLES \
    --validation_log_type                       $VALIDATION_LOG_TYPE \
    --apply_qwen_2_5_vl_pos_emb                 $APPLY_QWEN_2_5_VL_POS_EMB \
    --apply_chat_template                       $APPLY_CHAT_TEMPLATE
