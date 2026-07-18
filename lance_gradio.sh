#!/bin/bash
# Launch the Lance Gradio UI with one model sharded across the box's GPUs
# (model-parallel), matching the low-RAM / multi-small-GPU CLI path.
#
# IMPORTANT: the container (podman_run) must publish the UI port, e.g. add
#   -p 7860:7860
# to the `podman run` line, since the default podman_run publishes nothing.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# GPUs to shard the single model across. Must be contiguous 0..N-1 (the device_map
# uses logical cuda:0..N-1 with the entry/exit on cuda:0). To use specific physical
# cards, set CUDA_VISIBLE_DEVICES and keep LANCE_GPUS=0,1,...,N-1.
LANCE_GPUS=${LANCE_GPUS:-0,1,2,3,4}
SERVER_NAME=${SERVER_NAME:-0.0.0.0}   # bind all interfaces so a host browser can reach it
SERVER_PORT=${SERVER_PORT:-7860}

export LANCE_GPUS
# expandable_segments:True (for the streaming load) is set by lance_gradio.py itself
# via os.environ.setdefault before torch initializes CUDA.

echo "================================================"
echo "Lance Gradio (sharded across GPUs: ${LANCE_GPUS})"
echo "Serving on ${SERVER_NAME}:${SERVER_PORT}"
echo "================================================"

python lance_gradio.py \
    --gpus "$LANCE_GPUS" \
    --server-name "$SERVER_NAME" \
    --server-port "$SERVER_PORT" \
    "$@"
