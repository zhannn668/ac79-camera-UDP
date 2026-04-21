#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
杰理 AC79 / WL82 CTP 调试版客户端 / 服务端

这版重点增强：
1. 同时支持 client / server，便于先用 Windows 调试，再迁移到 RK。
2. 增加十六进制收发打印，能看到每一段原始字节。
3. 增加流式重组与重同步，能处理 TCP 半包 / 粘包 / 前导垃圾 / 尾部 0 填充。
4. 增加日志文件输出，便于复盘到底卡在哪一步。
5. 增加默认握手序列，一键测试 KEEP_ALIVE_INTERVAL / APP_ACCESS / DATE_TIME / OPEN_RT_STREAM。

CTP 应用层帧格式（按当前已知实现）：
    b"CTP:" + topic_len(2B) + topic + content_len(4B) + content

注意：
- 默认按 little endian 打包长度字段；如完全无回包，可尝试 --byteorder big。
- 这是 TCP 字节流调试器，不把一次 recv 当作一帧，而是按前缀和长度字段重组。
"""

from __future__ import annotations

import argparse
import json
import os
import select
import socket
import struct
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

MAGIC = b"CTP:"
DEFAULT_IP = "192.168.1.1"
DEFAULT_PORT = 3333
DEFAULT_LISTEN = "0.0.0.0"
MAX_TOPIC_LEN = 1024
MAX_CONTENT_LEN = 8 * 1024 * 1024


class Logger:
    def __init__(self, log_file: Optional[str] = None) -> None:
        self._fp = None
        self._lock = threading.Lock()
        if log_file:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._fp:
            self._fp.close()
            self._fp = None

    def log(self, msg: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        with self._lock:
            print(line)
            if self._fp:
                self._fp.write(line + "\n")
                self._fp.flush()


class Hex:
    @staticmethod
    def dump(data: bytes, width: int = 16) -> str:
        lines = []
        for i in range(0, len(data), width):
            chunk = data[i : i + width]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            asc_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:04X}  {hex_part:<{width * 3}}  {asc_part}")
        return "\n".join(lines)


class CtpCodec:
    def __init__(self, byteorder: str = "little") -> None:
        if byteorder not in ("little", "big"):
            raise ValueError("byteorder must be 'little' or 'big'")
        self.byteorder = byteorder
        self.u16 = "<H" if byteorder == "little" else ">H"
        self.u32 = "<I" if byteorder == "little" else ">I"

    def pack(self, topic: str, content: str) -> bytes:
        topic_b = topic.encode("utf-8")
        content_b = content.encode("utf-8")
        return b"".join(
            [
                MAGIC,
                struct.pack(self.u16, len(topic_b)),
                topic_b,
                struct.pack(self.u32, len(content_b)),
                content_b,
            ]
        )

    def pack_keep_alive(self) -> bytes:
        topic_b = b"CTP_KEEP_ALIVE"
        return b"".join(
            [
                MAGIC,
                struct.pack(self.u16, len(topic_b)),
                topic_b,
                struct.pack(self.u32, 0),
            ]
        )

    def try_parse_from_buffer(self, buf: bytearray) -> List[Tuple[bytes, str, str]]:
        frames: List[Tuple[bytes, str, str]] = []

        while True:
            idx = buf.find(MAGIC)
            if idx < 0:
                if len(buf) > len(MAGIC) - 1:
                    del buf[: -(len(MAGIC) - 1)]
                break

            if idx > 0:
                del buf[:idx]

            if len(buf) < 10:
                break

            topic_len = struct.unpack(self.u16, bytes(buf[4:6]))[0]
            if topic_len > MAX_TOPIC_LEN:
                del buf[0]
                continue

            need_topic_end = 6 + topic_len
            if len(buf) < need_topic_end + 4:
                break

            topic_b = bytes(buf[6:need_topic_end])
            content_len = struct.unpack(self.u32, bytes(buf[need_topic_end : need_topic_end + 4]))[0]
            if content_len > MAX_CONTENT_LEN:
                del buf[0]
                continue

            total_len = 4 + 2 + topic_len + 4 + content_len
            if len(buf) < total_len:
                break

            content_b = bytes(buf[need_topic_end + 4 : total_len])
            raw = bytes(buf[:total_len])
            try:
                topic = topic_b.decode("utf-8", errors="replace")
            except Exception:
                topic = repr(topic_b)
            try:
                content = content_b.decode("utf-8", errors="replace")
            except Exception:
                content = repr(content_b)

            frames.append((raw, topic, content))
            del buf[:total_len]

        return frames


class CtpDebugTool:
    def __init__(
        self,
        mode: str,
        host: str,
        port: int,
        listen: str,
        byteorder: str,
        recv_timeout: float,
        heartbeat_interval: Optional[float],
        pretty_json: bool,
        log_file: Optional[str],
    ) -> None:
        self.mode = mode
        self.host = host
        self.port = port
        self.listen = listen
        self.codec = CtpCodec(byteorder=byteorder)
        self.recv_timeout = recv_timeout
        self.heartbeat_interval = heartbeat_interval
        self.pretty_json = pretty_json
        self.logger = Logger(log_file)
        self.stop_event = threading.Event()
        self.sock: Optional[socket.socket] = None
        self.reader_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.rx_count = 0
        self.tx_count = 0

    def close(self) -> None:
        self.stop_event.set()
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self.logger.log("[OK] tool closed")
        self.logger.close()

    def connect_or_listen(self) -> None:
        if self.mode == "client":
            self.logger.log(f"[STAGE-1] connecting to {self.host}:{self.port}")
            s = socket.create_connection((self.host, self.port), timeout=5.0)
            s.settimeout(None)
            self.sock = s
            self.logger.log(f"[STAGE-2] connected to {self.host}:{self.port}")
            return

        self.logger.log(f"[STAGE-1] listening on {self.listen}:{self.port}")
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.listen, self.port))
        srv.listen(1)
        conn, addr = srv.accept()
        srv.close()
        conn.settimeout(None)
        self.sock = conn
        self.logger.log(f"[STAGE-2] peer connected from {addr[0]}:{addr[1]}")

    def start_reader(self) -> None:
        if self.reader_thread and self.reader_thread.is_alive():
            return
        self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader_thread.start()

    def start_heartbeat(self) -> None:
        if self.heartbeat_interval is None:
            return
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            return
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        assert self.heartbeat_interval is not None
        while not self.stop_event.is_set():
            time.sleep(self.heartbeat_interval)
            if self.stop_event.is_set():
                break
            try:
                self.send_keep_alive()
            except Exception as e:
                self.logger.log(f"[ERR] heartbeat failed: {e}")
                self.stop_event.set()
                break

    def _pretty_json(self, content: str) -> str:
        if not self.pretty_json:
            return content
        try:
            obj = json.loads(content)
            return json.dumps(obj, ensure_ascii=False, indent=2)
        except Exception:
            return content

    def _reader_loop(self) -> None:
        assert self.sock is not None
        buf = bytearray()
        while not self.stop_event.is_set():
            try:
                ready, _, _ = select.select([self.sock], [], [], self.recv_timeout)
                if not ready:
                    continue
                data = self.sock.recv(4096)
                if not data:
                    self.logger.log("[RX] socket closed by peer")
                    self.stop_event.set()
                    break

                self.logger.log(f"[RX-RAW] got {len(data)} bytes")
                for line in Hex.dump(data).splitlines():
                    self.logger.log(line)

                buf.extend(data)
                frames = self.codec.try_parse_from_buffer(buf)
                for raw, topic, content in frames:
                    self.rx_count += 1
                    self.logger.log("=" * 72)
                    self.logger.log(f"[RX-FRAME #{self.rx_count}] topic={topic}")
                    self.logger.log(f"[RX-FRAME #{self.rx_count}] raw_len={len(raw)}")
                    self.logger.log(f"[RX-FRAME #{self.rx_count}] content=")
                    for line in self._pretty_json(content).splitlines():
                        self.logger.log(line)
                    self.logger.log("=" * 72)
            except (OSError, ValueError) as e:
                self.logger.log(f"[ERR] reader loop: {e}")
                self.stop_event.set()
                break

    def _send_bytes(self, pkt: bytes, topic: str, content: str) -> None:
        if not self.sock:
            raise RuntimeError("socket not connected")
        self.sock.sendall(pkt)
        self.tx_count += 1
        self.logger.log("-" * 72)
        self.logger.log(f"[TX #{self.tx_count}] topic={topic}")
        self.logger.log(f"[TX #{self.tx_count}] frame_len={len(pkt)}")
        self.logger.log(f"[TX #{self.tx_count}] content={content}")
        for line in Hex.dump(pkt).splitlines():
            self.logger.log(line)
        self.logger.log("-" * 72)

    def send(self, topic: str, content: str) -> None:
        pkt = self.codec.pack(topic, content)
        self._send_bytes(pkt, topic, content)

    def send_get(self, topic: str) -> None:
        self.send(topic, json.dumps({"op": "GET"}, separators=(",", ":")))

    def send_keep_alive(self) -> None:
        pkt = self.codec.pack_keep_alive()
        self._send_bytes(pkt, "CTP_KEEP_ALIVE", "")

    def send_app_access(self, phone_type: int = 0, version: str = "1.0") -> None:
        payload = {"op": "PUT", "param": {"type": str(phone_type), "ver": version}}
        self.send("APP_ACCESS", json.dumps(payload, separators=(",", ":")))

    def send_date_time(self) -> None:
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        payload = {"op": "PUT", "param": {"date": now}}
        self.send("DATE_TIME", json.dumps(payload, separators=(",", ":")))

    def send_open_rt_stream(
        self, width: int = 640, height: int = 480, fps: int = 30, rate: int = 8000, fmt: int = 1
    ) -> None:
        payload = {
            "op": "PUT",
            "param": {
                "rate": str(rate),
                "w": str(width),
                "fps": str(fps),
                "h": str(height),
                "format": str(fmt),
            },
        }
        self.send("OPEN_RT_STREAM", json.dumps(payload, separators=(",", ":")))

    def send_default_sequence(
        self, width: int = 640, height: int = 480, fps: int = 30, rate: int = 8000, fmt: int = 1
    ) -> None:
        self.logger.log("[SEQ] sending default init sequence")
        self.send_get("KEEP_ALIVE_INTERVAL")
        time.sleep(0.2)
        self.send_app_access(phone_type=0, version="1.0")
        time.sleep(0.2)
        self.send_date_time()
        time.sleep(0.2)
        self.send_open_rt_stream(width=width, height=height, fps=fps, rate=rate, fmt=fmt)


def interactive_shell(tool: CtpDebugTool, width: int, height: int, fps: int, rate: int, fmt: int) -> None:
    help_text = """
可用命令：
  help
  seq                            发送默认调试序列
  get KEEP_ALIVE_INTERVAL
  get SD_STATUS
  get BAT_STATUS
  get UUID
  app                            发送 APP_ACCESS
  date                           发送 DATE_TIME
  keep                           发送一次 CTP_KEEP_ALIVE
  open [w h fps rate fmt]        发送 OPEN_RT_STREAM
  raw TOPIC JSON                 发送任意 topic + JSON
  quit
  sd:xxx                         发送 sd:xxx  指定播放杰里sd卡内音频
  sd_stop                        发送 sd_stop 停止播放音频
""".strip()
    tool.logger.log(help_text)

    while not tool.stop_event.is_set():
        try:
            line = input("ctp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line == "help":
            tool.logger.log(help_text)
            continue
        if line == "quit":
            break
        if line == "seq":
            tool.send_default_sequence(width=width, height=height, fps=fps, rate=rate, fmt=fmt)
            continue
        if line == "app":
            tool.send_app_access()
            continue
        if line == "date":
            tool.send_date_time()
            continue
        if line == "keep":
            tool.send_keep_alive()
            continue
        if line.startswith("get "):
            tool.send_get(line[4:].strip())
            continue
        if line.startswith("open"):
            parts = line.split()
            if len(parts) == 1:
                tool.send_open_rt_stream(width=width, height=height, fps=fps, rate=rate, fmt=fmt)
            elif len(parts) == 6:
                _, w, h, p_fps, p_rate, p_fmt = parts
                tool.send_open_rt_stream(
                    width=int(w), height=int(h), fps=int(p_fps), rate=int(p_rate), fmt=int(p_fmt)
                )
            else:
                tool.logger.log("用法: open [w h fps rate fmt]")
            continue
        if line.startswith("raw "):
            try:
                _, topic, payload = line.split(" ", 2)
            except ValueError:
                tool.logger.log("用法: raw TOPIC JSON")
                continue
            tool.send(topic, payload)
            continue
        
        import json

        # sd卡指定音频播放
        if line.startswith("sd:"):
            num = line[3:].strip()
            payload = {
                "op": "PUT",
                "param": {
                    "cmd": "play_sd_num",
                    "num": num,
                    "vol": "80"
                }
            }
            tool.send("GENERIC_CMD", json.dumps(payload, separators=(",", ":")))
            continue

        # 停止播放音频
        if line.strip() == "sd_stop":
            payload = {
                "op": "PUT",
                "param": {
                    "cmd": "stop_sd"
                }
            }
            tool.send("GENERIC_CMD", json.dumps(payload, separators=(",", ":")))
            continue



        tool.logger.log("未知命令，输入 help 查看帮助。")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="杰理 CTP 调试版客户端 / 服务端")
    p.add_argument("--mode", choices=["client", "server"], default="client")
    p.add_argument("--host", default=DEFAULT_IP, help="client 模式连接的设备 IP")
    p.add_argument("--listen", default=DEFAULT_LISTEN, help="server 模式监听地址")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="CTP 端口，默认 3333")
    p.add_argument(
        "--byteorder",
        choices=["little", "big"],
        default="little",
        help="长度字段字节序，默认 little；如无回包可试 big",
    )
    p.add_argument("--heartbeat", type=float, default=None, help="心跳周期（秒），例如 10")
    p.add_argument("--no-pretty", action="store_true", help="收到 JSON 时不美化输出")
    p.add_argument("--auto-seq", action="store_true", help="连接后自动发送默认调试序列")
    p.add_argument("--log-file", default=None, help="把所有输出同时保存到文件")
    p.add_argument("--recv-timeout", type=float, default=1.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rate", type=int, default=8000)
    p.add_argument("--format", type=int, default=1, choices=[0, 1], help="0=JPEG, 1=H264")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    tool = CtpDebugTool(
        mode=args.mode,
        host=args.host,
        port=args.port,
        listen=args.listen,
        byteorder=args.byteorder,
        recv_timeout=args.recv_timeout,
        heartbeat_interval=args.heartbeat,
        pretty_json=not args.no_pretty,
        log_file=args.log_file,
    )

    try:
        tool.connect_or_listen()
        tool.start_reader()
        tool.start_heartbeat()

        if args.auto_seq:
            tool.send_default_sequence(
                width=args.width,
                height=args.height,
                fps=args.fps,
                rate=args.rate,
                fmt=args.format,
            )

        interactive_shell(
            tool,
            width=args.width,
            height=args.height,
            fps=args.fps,
            rate=args.rate,
            fmt=args.format,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        print(f"[FATAL] {e}")
        return 1
    finally:
        tool.close()


if __name__ == "__main__":
    sys.exit(main())
