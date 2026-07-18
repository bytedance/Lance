#!/bin/bash

find_available_port() {
    local start_port="${1:-6666}"
    local end_port="${2:-8888}"

    python3 - "$start_port" "$end_port" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
end_port = int(sys.argv[2])

for port in range(start_port, end_port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", port))
        sock.close()
        print(port)
        raise SystemExit(0)
    except OSError:
        continue

print(start_port)
PY
}


lance_setup_common_env() {
    export EXP_HW_20250819="${EXP_HW_20250819:-False}"
    echo "EXP_HW_20250819: $EXP_HW_20250819"

    export POSITION_EMBEDDING_3D_VERSION="${POSITION_EMBEDDING_3D_VERSION:-v2}"
    echo "(shell) POSITION_EMBEDDING_3D_VERSION: $POSITION_EMBEDDING_3D_VERSION"

    # Default to async CUDA execution for benchmark/inference throughput.
    # Override with CUDA_LAUNCH_BLOCKING=1 only when debugging kernel failures.
    export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
    export NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-900}"

    # The streaming load in inference_lance.py allocates thousands of small-to-large
    # tensors onto each GPU as it walks the checkpoint. The default caching allocator
    # fragments under that pattern hard enough that a 1.2 GB tensor can fail to
    # allocate on a card that still has plenty of total free VRAM. expandable_segments
    # coalesces freed regions and lets large allocations grow into them. Required for
    # the 5×3060 sharded load to succeed.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
}


lance_setup_distributed_env() {
    local num_gpus="${1:-1}"
    local default_main_process_port
    local has_explicit_main_process_port=0

    NUM_GPUS="$num_gpus"

    if [ -n "$MAIN_PROCESS_PORT" ]; then
        has_explicit_main_process_port=1
    fi

    if [ -n "${ARNOLD_WORKER_NUM:-}" ]; then
        echo "使用平台分布式环境"
        NUM_MACHINES="${NUM_MACHINES:-$ARNOLD_WORKER_NUM}"
        MACHINE_RANK="${MACHINE_RANK:-${ARNOLD_ID:-0}}"
        MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
        default_main_process_port="${ARNOLD_WORKER_0_PORT:-6666}"

        if [ "$has_explicit_main_process_port" -eq 1 ]; then
            :
        elif [ "${NUM_MACHINES}" = "1" ]; then
            MAIN_PROCESS_PORT="$(find_available_port "$default_main_process_port" "$((default_main_process_port + 500))")"
        else
            MAIN_PROCESS_PORT="$default_main_process_port"
            echo "多机任务使用平台 rendezvous 端口: $MAIN_PROCESS_PORT"
        fi
    else
        echo "使用本地或显式配置的分布式环境"
        NUM_MACHINES="${NUM_MACHINES:-1}"
        MACHINE_RANK="${MACHINE_RANK:-0}"
        MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-127.0.0.1}"
        default_main_process_port=6666

        if [ "$has_explicit_main_process_port" -eq 1 ]; then
            :
        else
            MAIN_PROCESS_PORT="$(find_available_port "$default_main_process_port" "$((default_main_process_port + 500))")"
        fi
    fi

    TOTAL_RANK=$((NUM_MACHINES * NUM_GPUS))

    export NUM_GPUS NUM_MACHINES MACHINE_RANK MAIN_PROCESS_IP MAIN_PROCESS_PORT TOTAL_RANK

    echo "NUM_MACHINES: $NUM_MACHINES"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "TOTAL_RANK: $TOTAL_RANK"
    echo "MACHINE_RANK: $MACHINE_RANK"
    echo "MAIN_PROCESS_IP: $MAIN_PROCESS_IP"
    echo "MAIN_PROCESS_PORT: $MAIN_PROCESS_PORT"
}


lance_setup_shard_env() {
    local num_shard="${1:-1}"

    NUM_SHARD="$num_shard"
    NUM_REPLICATE=$((TOTAL_RANK / NUM_SHARD))

    export NUM_SHARD NUM_REPLICATE

    echo "NUM_REPLICATE: $NUM_REPLICATE"
    echo "NUM_SHARD: $NUM_SHARD"
}
