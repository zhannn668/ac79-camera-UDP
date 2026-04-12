# ac79-camera-UDP

面向 **杰理 AC79 / WL82** 设备的 Linux 侧最小联调工程，用于在 **RK3588 / Ubuntu / 其他 Linux 主机** 上完成两件事：

1. 通过 **CTP** 与设备建立控制链路，发送 `APP_ACCESS`、`DATE_TIME`、`OPEN_RT_STREAM` 等控制指令。  
2. 通过 **UDP** 接收设备发出的实时图像流，并在本地进行 **显示 / 保存 / 调试日志输出**。

这个仓库适合下面几类场景：

- 想在 RK3588 上直接接入杰理相机视频流，而不是走手机 App。
- 想先在 Linux 或 Windows 上把协议链路调通，再迁移到板端。
- 想观察 CTP 十六进制收发、定位握手失败点。
- 想接收 UDP JPEG 图像，后续再接入自己的视觉推理模型。

---

## 项目结构

```text
ac79-camera-UDP/
├── README.md
└── jieli_linux_bundle/
    ├── .env.example           # 环境变量模板
    ├── README_LINUX.md        # Linux 侧原始说明
    ├── requirements.txt       # Python 依赖
    ├── setup_env.sh           # 创建虚拟环境并安装依赖
    ├── start_udp.sh           # 前台启动 UDP 图像接收器
    ├── start_ctp.sh           # 前台启动 CTP 调试客户端/服务端
    ├── start_all.sh           # 后台启动 UDP + 自动发起 CTP 打流
    ├── stop_all.sh            # 停止后台进程
    ├── jieli_min_ctp_client.py
    └── jieli_min_udp_client.py
```

---

## 这个仓库能做什么

### 1. 一键拉起 Linux 端接收链路

`start_all.sh` 会完成以下动作：

- 清理旧的 UDP / CTP 进程
- 后台启动 `jieli_min_udp_client.py`
- 自动向设备发送：
  - `app`
  - `date`
  - `open w h fps rate fmt`
- 把 UDP 和 CTP 日志分别写入 `logs/udp.log` 与 `logs/ctp.log` / `logs/ctp_console.log`

这意味着你在 RK3588 上执行一次脚本，就能快速验证“**控制链路是否打通**”以及“**图像流是否已经开始发送**”。

### 2. 作为 CTP 协议调试工具使用

`jieli_min_ctp_client.py` 不只是普通客户端，它还支持：

- `client / server` 双模式
- 十六进制收发打印
- TCP 半包 / 粘包的流式重组
- 默认握手序列一键发送
- 将日志保存到文件，便于回放分析

适合在协议尚未完全稳定时排查问题。

### 3. 作为 UDP JPEG 实时接收器使用

`jieli_min_udp_client.py` 会：

- 监听 UDP 端口
- 过滤指定设备 IP（可关闭）
- 依据分片头把 JPEG 帧重新组装
- 判断 JPEG SOI / EOI
- 选择性显示窗口
- 按设定频率保存图片到本地
- 输出帧率、收包和清理信息

如果你后续要做目标检测、状态识别、多模态输入，这个脚本可以作为你自己的推理入口。

---

## 快速开始

### 1. 进入目录

```bash
cd jieli_linux_bundle
```

### 2. 初始化环境

```bash
./setup_env.sh
```

这个脚本会：

- 检查 `python3`
- 创建 `.venv`
- 安装依赖
- 自动生成 `.env`

### 3. 修改配置

首次运行后会生成 `.env`，按需修改：

```bash
cp .env.example .env
```

默认配置如下：

```env
# 设备与端口
DEVICE_IP=192.168.1.1
CTP_PORT=3333
UDP_PORT=2224

# 心跳与日志
HEARTBEAT=10
LOG_DIR=logs

# UDP 接收参数
SAVE_DIR=udp_frames
SAVE_EVERY=60
SHOW_WINDOW=1
VERBOSE_UDP=0
NO_FILTER=0
CLEANUP_TIMEOUT=3.0

# CTP 客户端模式
CTP_MODE=client
LISTEN_IP=0.0.0.0

# Python / venv
PYTHON_BIN=python3
VENV_DIR=.venv
```

### 4. 一键启动

```bash
./start_all.sh
```

启动成功后，你可以查看：

```bash
tail -f logs/udp.log
```

和：

```bash
tail -f logs/ctp_console.log
```

### 5. 停止所有后台进程

```bash
./stop_all.sh
```

---

## 推荐联调顺序

如果你的目标是“先打通链路，再稳定推流”，建议按这个顺序来：

### 方案 A：最省事的一键方式

```bash
cd jieli_linux_bundle
./setup_env.sh
./start_all.sh
```

`start_all.sh` 会自动发送：

```text
app
date
open 640 480 20 8000 0
```

其中：

- `640 480`：分辨率
- `20`：帧率
- `8000`：码率/速率参数
- `0`：格式（通常表示 JPEG）

### 方案 B：分步骤排查

先启动 UDP：

```bash
./start_udp.sh
```

再单独启动 CTP：

```bash
./start_ctp.sh
```

进入 `ctp>` 提示符后，依次输入：

```text
app
date
open 640 480 20 8000 0
```

这样你能更清楚地看到是：

- TCP/CTP 没连上
- 握手指令没收到回包
- 还是 UDP 图像流没有发出

---

## 脚本说明

### `setup_env.sh`

环境初始化脚本。

功能：

- 创建虚拟环境 `.venv`
- 安装 `numpy`、`opencv-python`
- 自动复制 `.env.example -> .env`
- 给 `.sh` / `.py` 文件补执行权限

### `start_udp.sh`

前台运行 UDP 图像接收器。

常见用途：

- 单独验证 UDP 端口是否能收到数据
- 调试窗口显示和图片保存
- 后续替换成自己的推理主程序

### `start_ctp.sh`

前台运行 CTP 工具。

支持：

- `client` 模式：主动连接设备
- `server` 模式：本地监听，等待对端连接

适合单独做协议调试。

### `start_all.sh`

推荐主入口。

功能：

- 自动清理旧进程
- 后台启动 UDP
- 自动通过 CTP 下发打流指令
- 保持 CTP 进程常驻
- 输出日志位置和停止方式

### `stop_all.sh`

停止 `udp.pid`、`ctp.pid` 记录的后台进程，并额外清理相关 Python 进程。

---

## CTP 调试命令

启动 `start_ctp.sh` 后，可以在交互界面使用这些命令：

```text
help
seq
get KEEP_ALIVE_INTERVAL
get SD_STATUS
get BAT_STATUS
get UUID
app
date
keep
open [w h fps rate fmt]
raw TOPIC JSON
quit
```

说明：

- `seq`：发送默认初始化序列
- `app`：发送 `APP_ACCESS`
- `date`：发送 `DATE_TIME`
- `keep`：发送一次心跳
- `open`：发送 `OPEN_RT_STREAM`
- `raw`：手工发任意 topic 和 JSON

如果你需要观察底层数据，CTP 工具会把收发内容按十六进制打印出来，并且尝试自动重组帧。

---

## UDP 接收器说明

UDP 图像接收器的核心能力：

- 默认监听 `0.0.0.0:2224`
- 默认只接收 `192.168.1.1` 发来的包
- 支持 `--no-filter` 关闭来源 IP 过滤
- 支持 `--no-window` 纯接收不显示
- 支持 `--save-dir` / `--save-every` 落盘
- 支持 `--cleanup-timeout` 清理超时残帧
- 支持 `--verbose` 输出统计信息

如果你不想弹窗，也不想落盘，只想把 JPEG 数据接到自己的推理程序，可以考虑：

```env
SHOW_WINDOW=0
SAVE_DIR=
SAVE_EVERY=1
```

然后把 `jieli_min_udp_client.py` 中 `handle_complete_frame()` 的处理逻辑改成：

- 内存解码
- 喂给模型
- 输出推理结果

---

## 日志与排错

### 查看 UDP 日志

```bash
tail -f logs/udp.log
```

### 查看 CTP 控制台日志

```bash
tail -f logs/ctp_console.log
```

### 常见问题 1：缺少 `python3-venv`

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
```

### 常见问题 2：端口被占用 / 已有旧进程残留

```bash
./stop_all.sh
```

然后再重新启动。

### 常见问题 3：没有图形界面

脚本会在没有 `DISPLAY` 时自动追加 `--no-window`。  
如果你本来就是在纯命令行或 SSH 环境下运行，这是正常行为。

### 常见问题 4：启动后没有画面

建议按顺序排查：

1. 设备和 Linux 主机是否在同一网络
2. `DEVICE_IP` 是否正确
3. `CTP_PORT=3333`、`UDP_PORT=2224` 是否与设备端一致
4. `app -> date -> open` 是否真正发送成功
5. `logs/ctp_console.log` 中是否有回包
6. `logs/udp.log` 中是否有收包记录

---

## 适合二次开发的方向

这个仓库已经提供了比较好的最小化基础，后续可以往这些方向扩展：

### 1. 接入 RK3588 本地视觉推理

在 `jieli_min_udp_client.py` 中把 JPEG 解码后的图像直接送给：

- YOLO 检测
- 区域占用判断
- 异常事件识别
- 自定义状态机

### 2. 用 Windows 先调协议，再迁移到 RK

由于 CTP 工具支持 `client / server` 双模式，并且带十六进制日志，非常适合先在 PC 端验证协议，再搬到板端。

### 3. 扩展更多控制指令

除了当前的 `APP_ACCESS`、`DATE_TIME`、`OPEN_RT_STREAM`，你还可以继续补：

- SD 卡相关控制
- 音频控制
- 参数读取 / 设置
- 拍照 / 停流 / 状态查询

---

## 适用场景总结

这个项目不是一个“完整产品”，而是一个**很适合做协议验证与链路打通的最小工程**。  
如果你的目标是：

- 在 RK3588 上替代手机 App 接杰理图像流
- 把 CTP + UDP 链路先跑通
- 为后续 AI 推理、监控、事件检测做输入层

那么这个仓库就是一个很好的起点。


