# 杰理 Linux 一键启动包

## 包含内容
- `setup_env.sh`：创建虚拟环境、安装依赖、补齐执行权限
- `start_udp.sh`：启动 UDP JPEG 接收器
- `start_ctp.sh`：启动 CTP 控制客户端
- `start_all.sh`：后台启动 UDP，前台启动 CTP
- `stop_all.sh`：停止后台 UDP 和相关 Python 进程
- `.env.example`：可改设备 IP、端口、保存目录、窗口显示等

## 最快使用方式
```bash
cd jieli_linux_bundle
./setup_env.sh
cp .env.example .env
# 按需改 .env
./start_all.sh
```

## 推荐交互顺序
启动 `start_all.sh` 后，在 CTP 控制窗口里输入：
```text
app
date
open 640 480 20 8000 0
```

## 说明
- `start_all.sh` 会把 UDP 接收器放到后台，日志写到 `logs/udp.log`
- 如果 Linux 没有图形界面，脚本会自动加 `--no-window`
- 如果你只想做推理，不弹窗也不落盘，可以在 `.env` 里设置：
```bash
SHOW_WINDOW=0
SAVE_DIR=
SAVE_EVERY=1
```
  然后自行把 `jieli_min_udp_client.py` 改成内存解码后直接喂推理模型

## 常见问题
### 1. 提示没有 python3-venv
先安装：
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

### 2. 提示端口被占用
说明已有一个 UDP 接收器在跑：
```bash
./stop_all.sh
```
然后再重启。

### 3. 后台 UDP 有没有真的起来
```bash
tail -f logs/udp.log
```
