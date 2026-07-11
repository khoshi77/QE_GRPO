#!/bin/bash
# mpirun が各ノードで 1 プロセスずつ起動する想定のエントリ。
# rank0 = Ray head + main_ppo 実行、rank>=1 = Ray worker (head が止まるまで block)。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_env.sh"

RANK=${OMPI_COMM_WORLD_RANK}
WORLD_SIZE=${OMPI_COMM_WORLD_SIZE}
HEAD_IP=${HEAD_IP:?HEAD_IP not set}
HEAD_PORT=${HEAD_PORT:-6379}

NUM_CPUS=${NUM_CPUS:-32}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}

echo "[rank ${RANK}/$((WORLD_SIZE-1))] host=$(hostname) HEAD_IP=${HEAD_IP}:${HEAD_PORT}"

# 念のため古い Ray を片付け
ray stop --force 2>/dev/null || true

if [ "${RANK}" = "0" ]; then
    # ---- Ray HEAD ----
    ray start --head \
        --node-ip-address="${HEAD_IP}" \
        --port="${HEAD_PORT}" \
        --num-cpus="${NUM_CPUS}" \
        --num-gpus="${NUM_GPUS_PER_NODE}" \
        --disable-usage-stats

    # 全 worker が join するまで待機
    echo "[head] waiting for ${WORLD_SIZE} nodes to join..."
    for i in $(seq 1 60); do
        n=$(ray list nodes 2>/dev/null | grep -c ALIVE || true)
        echo "[head] alive nodes: ${n}/${WORLD_SIZE}"
        if [ "${n}" -ge "${WORLD_SIZE}" ]; then
            break
        fi
        sleep 5
    done

    # 訓練本体を head 上で起動。RUN_SCRIPT 未指定なら gsm8k 版を使う。
    set +e
    bash "${SCRIPT_DIR}/${RUN_SCRIPT:-run_grpo_qwen3_4b.sh}"
    TRAIN_RC=$?
    set -e

    # クラスタ停止 → worker の `ray start --block` も解放される
    ray stop --force 2>/dev/null || true
    exit ${TRAIN_RC}
else
    # ---- Ray WORKER ----
    sleep 15  # head が listen し始めるまでの猶予
    ray start \
        --address="${HEAD_IP}:${HEAD_PORT}" \
        --num-cpus="${NUM_CPUS}" \
        --num-gpus="${NUM_GPUS_PER_NODE}" \
        --block
fi
