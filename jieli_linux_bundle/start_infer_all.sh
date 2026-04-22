#!/usr/bin/env bash
set -euo pipefail

# 功能：
# 1. 读取 .env 配置（如果有）
# 2. 后台启动 CTP，自动发送 app/date/open，打开杰理 JPEG 视频流
# 3. 前台启动 UDP 接收 + RKNN 推理 + LCD 显示
#
# 说明：
# - 这里不再依赖 XAUTHORITY，因为你当前系统里没有这个文件。
# - 只导出 DISPLAY。若 VSCode 远程终端仍无法显示到 LCD，
#   需要在 RK3588 本机 LCD 桌面终端执行一次：
#       xhost +SI:localuser:elf
#
# 用法：
#   chmod +x ./start_infer_all_fixed.sh
#   ./start_infer_all_fixed.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 读取 .env
if [[ -f "$SCRIPT_DIR/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ----------------------------
# 基础参数
# ----------------------------
DEVICE_IP="${DEVICE_IP:-192.168.1.1}"
UDP_PORT="${UDP_PORT:-2224}"
CTP_PORT="${CTP_PORT:-3333}"
HEARTBEAT="${HEARTBEAT:-10}"

# 杰理 OPEN_RT_STREAM 参数
OPEN_W="${OPEN_W:-640}"
OPEN_H="${OPEN_H:-480}"
OPEN_FPS="${OPEN_FPS:-20}"
OPEN_RATE="${OPEN_RATE:-8000}"
OPEN_FMT="${OPEN_FMT:-0}"   # 0=JPEG, 1=H264

# 显示参数
DISPLAY_NUM="${DISPLAY_NUM:-:0.0}"
DISPLAY_W="${DISPLAY_W:-1280}"
DISPLAY_H="${DISPLAY_H:-800}"

# 模型路径：优先用 .env 里的；若没写，就使用你给出的默认路径
MODEL_PATH="${MODEL_PATH:-/home/elf/work/jieli_linux_bundle/model/best.rknn}"
LABELS_PATH="${LABELS_PATH:-/home/elf/work/jieli_linux_bundle/model/labels.txt}"

# 推理阈值
OBJ_THRESH="${OBJ_THRESH:-0.25}"
NMS_THRESH="${NMS_THRESH:-0.45}"

# 日志目录
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

# Python 与脚本路径
PYTHON_BIN="python3"
if [[ -f "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
fi

CTP_SCRIPT="${CTP_SCRIPT:-$SCRIPT_DIR/jieli_min_ctp_client.py}"
INFER_SCRIPT="${INFER_SCRIPT:-$SCRIPT_DIR/jieli_rknn_udp_infer.py}"

# ----------------------------
# 基本检查
# ----------------------------
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "[ERR] 模型文件不存在: $MODEL_PATH"
    exit 1
fi

if [[ -n "$LABELS_PATH" && ! -f "$LABELS_PATH" ]]; then
    echo "[ERR] 标签文件不存在: $LABELS_PATH"
    exit 1
fi

# 导出 DISPLAY，子进程（OpenCV / Qt）会继承这个变量
export DISPLAY="$DISPLAY_NUM"

echo "[INFO] DISPLAY=$DISPLAY"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] LABELS_PATH=$LABELS_PATH"

# ----------------------------
# 清理旧进程
# ----------------------------
pkill -f "jieli_rknn_udp_infer.py" 2>/dev/null || true
pkill -f "jieli_min_ctp_client.py" 2>/dev/null || true
pkill -f "jieli_min_udp_client.py" 2>/dev/null || true
pkill -f "tail -f /dev/null" 2>/dev/null || true
rm -f "$SCRIPT_DIR/ctp.pid" "$SCRIPT_DIR/udp.pid"

# ----------------------------
# 后台启动 CTP
# ----------------------------
echo "[INFO] 后台启动 CTP，自动发送 app/date/open..."
(
    {
        sleep 1
        echo "app"
        sleep 1
        echo "date"
        sleep 1
        echo "open $OPEN_W $OPEN_H $OPEN_FPS $OPEN_RATE $OPEN_FMT"
        # 保持 stdin 不结束，否则 CTP 进程会退出
        tail -f /dev/null
    } | "$PYTHON_BIN" "$CTP_SCRIPT" \
        --mode client \
        --host "$DEVICE_IP" \
        --port "$CTP_PORT" \
        --heartbeat "$HEARTBEAT" \
        --log-file "$LOG_DIR/ctp.log"
) > "$LOG_DIR/ctp_console.log" 2>&1 &
CTP_PID=$!
echo "$CTP_PID" > "$SCRIPT_DIR/ctp.pid"

sleep 2
if ! kill -0 "$CTP_PID" 2>/dev/null; then
    echo "[ERR] CTP 启动失败，请查看日志：$LOG_DIR/ctp_console.log"
    exit 1
fi

echo "[OK] CTP 已启动，PID=$CTP_PID"

cleanup() {
    kill "$CTP_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ----------------------------
# 前台启动推理
# ----------------------------
echo "[INFO] 前台启动 UDP 接收 + RKNN 推理 + LCD 显示..."

ARGS=(
    --model "$MODEL_PATH"
    --device-ip "$DEVICE_IP"
    --port "$UDP_PORT"
    --fullscreen
    --display-size "$DISPLAY_W" "$DISPLAY_H"
    --obj-thresh "$OBJ_THRESH"
    --nms-thresh "$NMS_THRESH"
)



if [[ -n "$LABELS_PATH" ]]; then
    ARGS+=(--labels "$LABELS_PATH")
fi

# 如果你模型需要 BGR 输入，则在 .env 里写 BGR_INPUT=1
if [[ "${BGR_INPUT:-0}" == "1" ]]; then
    ARGS+=(--bgr-input)
fi

# 如果你只想用单核，则在 .env 里写 SINGLE_CORE=1
if [[ "${SINGLE_CORE:-0}" == "1" ]]; then
    ARGS+=(--single-core)
fi

# 如果需要详细日志，则在 .env 里写 VERBOSE_INFER=1
if [[ "${VERBOSE_INFER:-0}" == "1" ]]; then
    ARGS+=(--verbose)
fi

exec env DISPLAY="$DISPLAY" "$PYTHON_BIN" "$INFER_SCRIPT" "${ARGS[@]}"
