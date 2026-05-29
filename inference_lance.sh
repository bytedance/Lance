#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/benchmarks/sample_env.sh"

# ========================= Inference Parameters =========================
NUM_GPUS=${NUM_GPUS:-1}

TASK_NAME=${TASK_NAME:-x2t_image} # t2i | image_edit | t2v | i2v | video_edit | x2t_image | x2t_video

VALIDATION_NUM_TIMESTEPS=${VALIDATION_NUM_TIMESTEPS:-30}
VALIDATION_TIMESTEP_SHIFT=${VALIDATION_TIMESTEP_SHIFT:-3.5}
VALIDATION_DATA_SEED=${VALIDATION_DATA_SEED:-42}
CFG_TEXT_SCALE=${CFG_TEXT_SCALE:-4.0}
USE_KVCACHE=${USE_KVCACHE:-true}
ENHANCE_PROMPT=${ENHANCE_PROMPT:-false}

NUM_FRAMES=${NUM_FRAMES:-50}             # max: 121 frames, unused for image tasks
VIDEO_HEIGHT=${VIDEO_HEIGHT:-768}        # unused for editing
VIDEO_WIDTH=${VIDEO_WIDTH:-768}          # unused for editing
RESOLUTION=${RESOLUTION:-"video_480p"}   # image_768res | video_480p
TEXT_TEMPLATE=${TEXT_TEMPLATE:-true}

MODEL_PATH=${MODEL_PATH:-"downloads/Lance_3B_Video"}
CONFIG_PATH=${CONFIG_PATH:-""}

# ========================= Command-line Arguments =========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --NUM_GPUS) NUM_GPUS="$2"; shift 2 ;;
        --TASK_NAME) TASK_NAME="$2"; shift 2 ;;
        --MODEL_PATH) MODEL_PATH="$2"; shift 2 ;;
        --CONFIG_PATH) CONFIG_PATH="$2"; shift 2 ;;

        --VALIDATION_NUM_TIMESTEPS) VALIDATION_NUM_TIMESTEPS="$2"; shift 2 ;;
        --VALIDATION_TIMESTEP_SHIFT) VALIDATION_TIMESTEP_SHIFT="$2"; shift 2 ;;
        --VALIDATION_DATA_SEED) VALIDATION_DATA_SEED="$2"; shift 2 ;;
        --CFG_TEXT_SCALE) CFG_TEXT_SCALE="$2"; shift 2 ;;
        --USE_KVCACHE) USE_KVCACHE="$2"; shift 2 ;;
        --ENHANCE_PROMPT) ENHANCE_PROMPT="$2"; shift 2 ;;

        --NUM_FRAMES) NUM_FRAMES="$2"; shift 2 ;;
        --VIDEO_HEIGHT) VIDEO_HEIGHT="$2"; shift 2 ;;
        --VIDEO_WIDTH) VIDEO_WIDTH="$2"; shift 2 ;;
        --RESOLUTION) RESOLUTION="$2"; shift 2 ;;
        --TEXT_TEMPLATE) TEXT_TEMPLATE="$2"; shift 2 ;;
        --SAVE_PATH_GEN) SAVE_PATH_GEN="$2"; shift 2 ;;

        -h|--help)
            echo "Usage: bash inference_lance_my.sh [OPTIONS]"
            echo ""
            echo "Example:"
            echo "  bash inference_lance_my.sh --TASK_NAME t2i --MODEL_PATH downloads/Lance_3B --RESOLUTION image_768res"
            echo "  bash inference_lance_my.sh --TASK_NAME image_edit --CONFIG_PATH config.json"
            echo "  bash inference_lance_my.sh --TASK_NAME t2v --ENHANCE_PROMPT true"
            echo "  bash inference_lance_my.sh --TASK_NAME i2v --ENHANCE_PROMPT true"
            exit 0
            ;;

        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage."
            exit 1
            ;;
    esac
done

# ========================= Auto-generated Paths =========================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
KVCACHE_TAG=""
if [ "$USE_KVCACHE" = "true" ]; then
    KVCACHE_TAG="_kvcache"
fi

DEFAULT_SAVE_PATH_GEN="results/${TASK_NAME}_sample_ts${VALIDATION_NUM_TIMESTEPS}_tts${VALIDATION_TIMESTEP_SHIFT}_seed${VALIDATION_DATA_SEED}_cfg${CFG_TEXT_SCALE}${KVCACHE_TAG}_${TIMESTAMP}"
SAVE_PATH_GEN=${SAVE_PATH_GEN:-"$DEFAULT_SAVE_PATH_GEN"}


if [ -z "$MODEL_PATH" ]; then
    echo "Error: please set MODEL_PATH manually in the configuration section at the top of this script."
    exit 1
fi

# ============================== Environment and Distributed Setup ==============================
lance_setup_common_env
lance_setup_distributed_env "$NUM_GPUS"
lance_setup_shard_env 1

# ========================= Show Task Configuration =========================
echo "================================================"
echo "Lance Inference"
echo "================================================"
echo "Task: ${TASK_NAME}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Save path: ${SAVE_PATH_GEN}"
echo "Resolution: ${VIDEO_HEIGHT}x${VIDEO_WIDTH}"
echo "Output frames: ${NUM_FRAMES}"
echo "Model path: ${MODEL_PATH}"
if [ -n "$CONFIG_PATH" ]; then
    echo "Config path: ${CONFIG_PATH}"
else
    echo "Config path: task default"
fi
echo ""
echo "Key parameters:"
echo "  - validation_num_timesteps: ${VALIDATION_NUM_TIMESTEPS}"
echo "  - validation_timestep_shift: ${VALIDATION_TIMESTEP_SHIFT}"
echo "  - validation_data_seed: ${VALIDATION_DATA_SEED}"
echo "  - cfg_text_scale: ${CFG_TEXT_SCALE}"
echo "  - num_frames: ${NUM_FRAMES}"
echo "  - use_KVcache: ${USE_KVCACHE}"
echo "  - enhance_prompt: ${ENHANCE_PROMPT}"
echo "================================================"
echo ""

# ============================== Run Inference ==============================
CONFIG_ARGS=()
if [ -n "$CONFIG_PATH" ]; then
    CONFIG_ARGS=(--val_dataset_config_file "$CONFIG_PATH")
fi

accelerate launch \
    --num_machines          $NUM_MACHINES \
    --num_processes         $TOTAL_RANK \
    --machine_rank          $MACHINE_RANK \
    --main_process_ip       $MAIN_PROCESS_IP \
    --main_process_port     $MAIN_PROCESS_PORT \
    --mixed_precision       bf16 \
    inference_lance.py \
    --model_path            "$MODEL_PATH" \
    --vit_type              qwen_2_5_vl_original \
    --llm_qk_norm           true \
    --llm_qk_norm_und       true \
    --llm_qk_norm_gen       true \
    --tie_word_embeddings   false \
    --validation_num_timesteps $VALIDATION_NUM_TIMESTEPS \
    --validation_timestep_shift $VALIDATION_TIMESTEP_SHIFT \
    --copy_init_moe         true \
    --max_num_frames        121 \
    --max_latent_size       64 \
    --latent_patch_size     1 1 1 \
    --visual_und            true \
    --visual_gen            true \
    --vae_model_type        wan \
    --apply_qwen_2_5_vl_pos_emb true \
    --apply_chat_template   false \
    --cfg_type              0 \
    --validation_data_seed  $VALIDATION_DATA_SEED \
    --video_height          $VIDEO_HEIGHT \
    --video_width           $VIDEO_WIDTH \
    --num_frames            $NUM_FRAMES \
    --task                  $TASK_NAME \
    --save_path_gen         "$SAVE_PATH_GEN" \
    --resolution            "$RESOLUTION" \
    --text_template         "$TEXT_TEMPLATE" \
    --cfg_text_scale        $CFG_TEXT_SCALE \
    --use_KVcache           "$USE_KVCACHE" \
    --enhance_prompt        "$ENHANCE_PROMPT" \
    "${CONFIG_ARGS[@]}"

echo ""
echo "================================================"
echo "Done! Results: ${SAVE_PATH_GEN}"
echo "================================================"
