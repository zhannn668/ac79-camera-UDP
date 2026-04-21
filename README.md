# Jieli AC79 / WL82 Linux Bundle

> 用于在 Linux / RK3588 端完成 **CTP 控制调试**、**UDP JPEG 实时流接收**、**RKNN 推理显示** 的一套启动脚本与 Python 工具。

## Features

- 支持连接杰理设备 `3333/TCP` 控制口，发送 `app / date / open` 等命令
- 支持接收杰理设备 `2224/UDP` 发出的 JPEG 视频流
- 支持本地窗口显示、按间隔保存 JPEG 帧
- 支持在 RK3588 上使用 `RKNNLite` 做目标检测并全屏显示到 LCD
- 提供一键启动脚本，适合先调链路、再接推理

---

## Project Structure

```text
.
├── jieli_min_ctp_client.py      # CTP 调试客户端/服务端
├── jieli_min_udp_client.py      # UDP JPEG 流接收器
├── jieli_rknn_udp_infer.py      # UDP JPEG + RKNNLite 推理显示
├── setup_env.sh                 # 创建 .venv 并安装依赖
├── start_ctp.sh                 # 前台启动 CTP
├── start_udp.sh                 # 前台启动 UDP 接收
├── start_all.sh                 # 后台 UDP + 后台 CTP 自动发 app/date/open
├── start_infer_all.sh           # 后台 CTP + 前台 RKNN 推理
├── stop_all.sh                  # 停止相关进程
├── requirements.txt             # Python 基础依赖
├── .env.example                 # 环境变量模板
└── README.md                    # 项目说明
```

---

## Supported Workflows

### 1. 只做链路调试
适合先确认：
- CTP 能不能连上
- `open` 命令有没有发出去
- UDP JPEG 流能不能收到
- 本地显示和保存是否正常

使用：

```bash
./start_all.sh
```

### 2. 直接做 RKNN 推理显示
适合在 RK3588 板端已经具备 `.rknn` 模型和 `rknnlite` 运行环境后使用。

使用：

```bash
./start_infer_all.sh
```

---

## Requirements

### System Packages

建议先安装：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

### Python Packages

基础依赖：

```txt
numpy
opencv-python
```

安装方式：

```bash
./setup_env.sh
```

> 注意：`requirements.txt` 目前只包含基础依赖。  
> 如果你要运行 `jieli_rknn_udp_infer.py`，还需要在 **目标 RK3588 板端** 安装与系统匹配的 `rknnlite` 运行环境。

---

## Quick Start

### Step 1: Prepare Environment

```bash
cd jieli_linux_bundle
./setup_env.sh
cp .env.example .env
```

### Step 2: Edit `.env`

按需修改设备 IP、端口、显示参数、模型路径等。

### Step 3: Start Debug Workflow

先跑普通链路调试：

```bash
./start_all.sh
```

查看日志：

```bash
tail -f logs/udp.log
tail -f logs/ctp_console.log
```

### Step 4: Start RKNN Inference

确认 JPEG 流没问题后，再切推理：

```bash
./stop_all.sh
./start_infer_all.sh
```

---

## `.venv` Notes

这一节很重要。

### 不要下载别人的 `.venv`

`.venv` 是**当前机器本地生成的 Python 虚拟环境**，不是项目源码的一部分。

不要这样做：
- 从别人电脑直接拷 `.venv`
- 从 x86 Linux 拷到 RK3588 ARM 板子
- 从 Windows 拷到 Linux
- 把 `.venv` 打包上传到 GitHub

这些做法很容易导致：
- Python 路径失效
- 动态库不匹配
- `ImportError`
- `ModuleNotFoundError`
- `wrong ELF class`
- `Illegal instruction`

### 正确做法

只下载项目源码，在**目标机器本地**创建 `.venv`：

```bash
./setup_env.sh
```

### RK3588 特别说明

如果涉及 `RKNNLite`：
- 一定要在 **板端** 安装对应版本运行库
- 不要把 PC 上创建好的 `.venv` 直接拷到板子上
- `.rknn` 模型、`rknnlite`、驱动版本要互相匹配

一句话：

> `.venv` 要在目标机现建，不要下载现成的。`

---

## Environment Variables

典型配置如下：

```bash
# device
DEVICE_IP=192.168.1.1
CTP_PORT=3333
UDP_PORT=2224

# CTP
HEARTBEAT=10
CTP_MODE=client
LISTEN_IP=0.0.0.0

# UDP
SAVE_DIR=udp_frames
SAVE_EVERY=60
SHOW_WINDOW=1
VERBOSE_UDP=0
NO_FILTER=0
CLEANUP_TIMEOUT=3.0

# Python
PYTHON_BIN=python3
VENV_DIR=.venv

# display
DISPLAY_NUM=:0.0
DISPLAY_W=1280
DISPLAY_H=800

# model
MODEL_PATH=/path/to/model.rknn
LABELS_PATH=/path/to/labels.txt
OBJ_THRESH=0.25
NMS_THRESH=0.45
BGR_INPUT=0
SINGLE_CORE=0
VERBOSE_INFER=0
```

---

## Usage

### Start CTP Only

```bash
./start_ctp.sh
```

进入交互后可手动输入：

```text
app
date
open 640 480 20 8000 0
```

### Start UDP Receiver Only

```bash
./start_udp.sh
```

### Start Full Debug Workflow

```bash
./start_all.sh
```

### Start Full Inference Workflow

```bash
./start_infer_all.sh
```

### Stop All

```bash
./stop_all.sh
```

---

## Recommended Debug Order

建议不要一上来就同时排查网络、CTP、UDP、JPEG 解码、RKNN 推理、LCD 显示。

更稳的顺序是：

1. `./setup_env.sh`
2. `./start_all.sh`
3. 看 `logs/udp.log` 和 `logs/ctp_console.log`
4. 确认 JPEG 流稳定
5. `./stop_all.sh`
6. `./start_infer_all.sh`

---

## Troubleshooting

### 1. `python3 -m venv` 失败

安装系统包：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

### 2. `.venv` 有但脚本还是跑不起来

最省事的办法是删掉重建：

```bash
rm -rf .venv
./setup_env.sh
```

### 3. 端口被占用

```bash
./stop_all.sh
```

### 4. 看不到窗口

先确认：

```bash
echo $DISPLAY
```

如果是远程终端启动图形界面失败，按你的板端用户执行一次：

```bash
xhost +SI:localuser:elf
```

### 5. 收不到视频流

重点检查：
- `DEVICE_IP` 是否正确
- 杰理端是否收到 `app / date / open`
- `OPEN_FMT` 是否为 `0`（JPEG）
- `UDP_PORT` 是否一致
- 板端防火墙是否拦截

### 6. 推理时报 `rknnlite` 错误

通常是：
- 板端没装 `rknnlite`
- 版本不匹配
- `.rknn` 模型和运行时不匹配
- 你把别的机器上的 `.venv` 拷过来了

---

## Recommended `.gitignore`

建议把下面这些加进 `.gitignore`：

```gitignore
.venv/
logs/
udp_frames/
*.pid
.env
```

如果 `.env` 里没有敏感信息，也可以只忽略真实 `.env`，保留 `.env.example`。

---

## Release Advice

如果你要把这个项目发给别人，建议只保留：

- 源码脚本：`*.py`、`*.sh`
- 配置模板：`.env.example`
- 依赖文件：`requirements.txt`
- 文档：`README.md`

不要带：

- `.venv/`
- `logs/`
- `udp_frames/`
- `*.pid`
- 运行时生成的临时文件

---

## Summary

这个项目的正确使用方式不是：

> 下载别人打包好的 `.venv` 然后直接运行

而是：

```bash
下载源码 -> 本机执行 ./setup_env.sh -> 修改 .env -> 先调链路 -> 再上推理
```

这样最稳，也最符合你现在这套脚本的设计逻辑。
