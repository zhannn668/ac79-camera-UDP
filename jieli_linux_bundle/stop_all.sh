#!/usr/bin/env bash
set -euo pipefail

# 功能：停止普通 UDP、CTP、推理版 UDP、以及保持 stdin 的 tail 进程

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for f in udp.pid ctp.pid; do
    if [[ -f "$f" ]]; then
        PID="$(cat "$f" || true)"
        if [[ -n "${PID:-}" ]] && kill -0 "$PID" 2>/dev/null; then
            echo "[INFO] 停止 $f 对应进程 PID=$PID"
            kill "$PID" 2>/dev/null || true
            sleep 1
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$f"
    fi
done

pkill -f "jieli_min_udp_client.py" 2>/dev/null || true
pkill -f "jieli_min_ctp_client.py" 2>/dev/null || true
pkill -f "jieli_rknn_udp_infer.py" 2>/dev/null || true
pkill -f "tail -f /dev/null" 2>/dev/null || true

echo "[OK] 停止完成。"
