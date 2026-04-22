#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
杰理 UDP JPEG 流 + RKNN 单输出/多输出 YOLO 检测 + LCD 显示

这版重点：
1. 明确按 NHWC 4 维输入喂给 RKNNLite，避免隐式格式转换。
2. 兼容单输出模型，例如 (1, 5, 8400) 这种 1 类检测模型。
3. 增加调试信息：打印输出 shape、score 范围、box 范围。
4. 支持 conf/iou/max_det/agnostic_nms 这些常见 YOLO 参数。

用法示例：
python3 jieli_rknn_udp_infer_debug.py \
    --model /path/to/best.rknn \
    --labels /path/to/labels.txt \
    --obj-thresh 0.25 \
    --nms-thresh 0.45 \
    --max-det 5 \
    --agnostic-nms
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np
from rknnlite.api import RKNNLite

PCM_TYPE_AUDIO = 0x01
JPEG_TYPE_VIDEO = 0x02
LAST_VIDEO_MARKER = 0x80
UDP_HEADER_LEN = 20

DFL_LEN = 16
STRIDES = (8, 16, 32)


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LetterBoxInfo:
    ratio: float
    pad_w: float
    pad_h: float
    new_w: int
    new_h: int


class FPSCounter:
    def __init__(self) -> None:
        self.last_time: Optional[float] = None
        self.fps = 0.0

    def update(self) -> float:
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else (0.9 * self.fps + 0.1 * inst)
        self.last_time = now
        return self.fps


class FrameState:
    def __init__(self, seq: int, frame_size: int, timestamp: int, media_type: int) -> None:
        self.seq = seq
        self.frame_size = frame_size
        self.timestamp = timestamp
        self.media_type = media_type
        self.buf = bytearray(frame_size)
        self.received = 0
        self.offsets: Set[int] = set()
        self.updated_at = time.time()

    def add_chunk(self, offset: int, payload: bytes, is_last: bool) -> None:
        end = offset + len(payload)
        if offset in self.offsets:
            return
        if offset < 0 or end > self.frame_size:
            return
        self.buf[offset:end] = payload
        self.offsets.add(offset)
        self.received += len(payload)
        self.updated_at = time.time()

    def is_complete(self) -> bool:
        return self.received >= self.frame_size

    def age(self) -> float:
        return time.time() - self.updated_at

    def to_bytes(self) -> bytes:
        return bytes(self.buf[: self.frame_size])


class YoloRknnDetector:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str],
        input_size: Tuple[int, int],
        obj_thresh: float,
        nms_thresh: float,
        max_det: int,
        agnostic_nms: bool,
        use_rgb: bool,
        use_all_cores: bool,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.labels = self._load_labels(labels_path)
        self.input_h, self.input_w = input_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.use_rgb = use_rgb
        self.verbose = verbose
        self.rknn = RKNNLite()
        self._logged_output_shapes = False
        self._logged_score_stats = False

        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed: ret={ret}, model={self.model_path}")

        if use_all_cores:
            ret = self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                log("[WARN] NPU_CORE_0_1_2 初始化失败，自动退回默认 init_runtime()")
                ret = self.rknn.init_runtime()
        else:
            ret = self.rknn.init_runtime()

        if ret != 0:
            raise RuntimeError(f"init_runtime failed: ret={ret}")

        log(f"[OK] RKNN 模型已加载: {self.model_path}")
        log(f"[INFO] 输入尺寸: {self.input_w}x{self.input_h}")
        log(f"[INFO] 类别数: {len(self.labels)}")

    def close(self) -> None:
        try:
            self.rknn.release()
        except Exception:
            pass

    @staticmethod
    def _load_labels(labels_path: Optional[str]) -> List[str]:
        if labels_path is None:
            return []
        path = Path(labels_path)
        if not path.exists():
            return []
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.labels):
            return self.labels[class_id]
        return f"class_{class_id}"

    def _letterbox(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, LetterBoxInfo]:
        h, w = image_bgr.shape[:2]
        ratio = min(self.input_w / w, self.input_h / h)
        new_w = int(round(w * ratio))
        new_h = int(round(h * ratio))

        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)

        pad_w = (self.input_w - new_w) / 2.0
        pad_h = (self.input_h - new_h) / 2.0
        left = int(round(pad_w - 0.1))
        top = int(round(pad_h - 0.1))
        canvas[top: top + new_h, left: left + new_w] = resized

        if self.use_rgb:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        return canvas, LetterBoxInfo(ratio, pad_w, pad_h, new_w, new_h)

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union

            inds = np.where(iou <= iou_thr)[0]
            order = order[inds + 1]

        return keep

    def _postprocess_single_output(
        self,
        output: np.ndarray,
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        兼容常见单输出：
        - (1, 5, 8400)  -> 单类，格式通常是 xywh + cls
        - (1, 84, 8400) -> 多类，格式通常是 xywh + cls...
        - (1, 8400, 5/84)
        """
        arr = np.asarray(output)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"单输出模型当前只支持 2 维主体张量，实际 shape={output.shape}")

        # 转成 [N, C]
        if arr.shape[0] < arr.shape[1]:
            arr = arr.T

        if arr.shape[1] < 5:
            raise ValueError(f"单输出张量列数异常，至少应为 5，实际 shape={arr.shape}")

        boxes_xywh = arr[:, :4].astype(np.float32)
        cls_scores = arr[:, 4:].astype(np.float32)

        # 输出调试
        if not self._logged_score_stats:
            log(f"[INFO] 单输出主体 shape(转置后): {arr.shape}")
            log(f"[INFO] box min/max: {boxes_xywh.min():.4f} / {boxes_xywh.max():.4f}")
            log(f"[INFO] score min/max(before): {cls_scores.min():.6f} / {cls_scores.max():.6f}")
            self._logged_score_stats = True

        # 自适应是否需要 sigmoid
        if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
            cls_scores = sigmoid(cls_scores)
            log("[INFO] 检测到 score 超出 [0,1]，已自动对 score 做 sigmoid")

        if cls_scores.shape[1] == 1:
            scores = cls_scores[:, 0]
            class_ids = np.zeros((arr.shape[0],), dtype=np.int32)
        else:
            class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = np.max(cls_scores, axis=1).astype(np.float32)

        if not self._logged_output_shapes:
            log(f"[INFO] score min/max(after): {scores.min():.6f} / {scores.max():.6f}")
            self._logged_output_shapes = True

        # 如果框是归一化坐标，缩放到输入尺寸
        if boxes_xywh.max() <= 2.0:
            boxes_xywh[:, [0, 2]] *= self.input_w
            boxes_xywh[:, [1, 3]] *= self.input_h
            log("[INFO] 检测到 box 范围较小，按归一化 xywh 处理并放大到输入尺寸")

        mask = scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 从 letterbox 输入图映射回原图
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] -= lb.pad_w
        boxes[:, [1, 3]] -= lb.pad_h
        boxes[:, :4] /= lb.ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []

        if self.agnostic_nms:
            keep = self._nms(boxes, scores, self.nms_thresh)
            keep = keep[: self.max_det]
            for k in keep:
                x1, y1, x2, y2 = boxes[k]
                results.append((int(class_ids[k]), float(scores[k]), (int(x1), int(y1), int(x2), int(y2))))
            return results

        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = self._nms(cls_boxes, cls_scores, self.nms_thresh)
            for k in keep:
                x1, y1, x2, y2 = cls_boxes[k]
                results.append((int(cls_id), float(cls_scores[k]), (int(x1), int(y1), int(x2), int(y2))))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[: self.max_det]

    def infer(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        blob, lb = self._letterbox(image_bgr)
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)

        outputs = self.rknn.inference(inputs=[blob], data_format=['nhwc'])
        if outputs is None:
            raise RuntimeError("rknn.inference() 返回了 None")

        if not hasattr(self, "_printed_shapes"):
            shapes = [tuple(np.asarray(o).shape) for o in outputs]
            log(f"[INFO] 模型输出数量: {len(outputs)}")
            log(f"[INFO] 模型输出 shape: {shapes}")
            self._printed_shapes = True

        if len(outputs) == 1:
            return self._postprocess_single_output(outputs[0], lb, image_bgr.shape[:2])

        raise ValueError(f"当前调试版主要针对单输出模型，你这个模型输出 {len(outputs)} 个，需另外对齐。")

    def draw(self, image_bgr: np.ndarray, results: List[Tuple[int, float, Tuple[int, int, int, int]]]) -> np.ndarray:
        vis = image_bgr.copy()
        for class_id, score, (x1, y1, x2, y2) in results:
            label = f"{self._class_name(class_id)} {score:.2f}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), (0, 255, 0), -1)
            cv2.putText(vis, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return vis


class JieliRknnUdpInfer:
    def __init__(
        self,
        bind_ip: str,
        bind_port: int,
        device_ip: Optional[str],
        cleanup_timeout: float,
        show_window: bool,
        fullscreen: bool,
        display_size: Optional[Tuple[int, int]],
        detector: YoloRknnDetector,
        verbose: bool,
    ) -> None:
        self.bind_ip = bind_ip
        self.bind_port = bind_port
        self.device_ip = device_ip
        self.cleanup_timeout = cleanup_timeout
        self.show_window = show_window
        self.fullscreen = fullscreen
        self.display_size = display_size
        self.detector = detector
        self.verbose = verbose

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.bind_ip, self.bind_port))
        self.sock.settimeout(1.0)

        self.frames: Dict[int, FrameState] = {}
        self.fps_counter = FPSCounter()
        self.window_name = "Jieli RKNN Infer"
        self.window_inited = False

        log(f"[INFO] UDP 监听: {self.bind_ip}:{self.bind_port}")
        if self.device_ip:
            log(f"[INFO] 仅接收设备 IP: {self.device_ip}")
        log(f"[INFO] 显示窗口: {'开' if self.show_window else '关'}")

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
        try:
            self.detector.close()
        except Exception:
            pass
        if self.show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        log("[OK] 程序已退出")

    def _init_window(self) -> None:
        if not self.show_window or self.window_inited:
            return
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.window_inited = True

    def _cleanup_stale_frames(self) -> None:
        stale_keys = [seq for seq, st in self.frames.items() if st.age() > self.cleanup_timeout]
        for seq in stale_keys:
            self.frames.pop(seq, None)

    def _handle_complete_frame(self, state: FrameState) -> bool:
        payload = state.to_bytes()
        if not payload.startswith(b"\xFF\xD8") or b"\xFF\xD9" not in payload:
            return False

        arr = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return False

        try:
            results = self.detector.infer(frame_bgr)
        except Exception as e:
            log(f"[ERR] 推理失败: {e}")
            return False

        vis = self.detector.draw(frame_bgr, results)
        fps = self.fps_counter.update()

        cv2.putText(vis, f"FPS: {fps:.1f}  DET: {len(results)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if self.display_size is not None:
            vis = cv2.resize(vis, self.display_size)

        if self.show_window:
            self._init_window()
            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return True
        return False

    def _parse_udp_packet(self, packet: bytes, addr: Tuple[str, int]) -> bool:
        if self.device_ip and addr[0] != self.device_ip:
            return False

        pos = 0
        plen = len(packet)
        while pos + UDP_HEADER_LEN <= plen:
            try:
                media_type, reserved, payload_len, seq, frame_size, offset, timestamp = struct.unpack_from(
                    "<BBHIIII", packet, pos
                )
            except struct.error:
                break

            pos += UDP_HEADER_LEN
            if payload_len == 0 or pos + payload_len > plen:
                break

            payload = packet[pos: pos + payload_len]
            pos += payload_len

            base_type = media_type & 0x7F
            if base_type != JPEG_TYPE_VIDEO:
                continue

            st = self.frames.get(seq)
            if st is None:
                st = FrameState(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)
                self.frames[seq] = st

            st.add_chunk(offset=offset, payload=payload, is_last=bool(media_type & LAST_VIDEO_MARKER))
            if st.is_complete():
                self.frames.pop(seq, None)
                if self._handle_complete_frame(st):
                    return True

        return False

    def run(self) -> int:
        try:
            while True:
                try:
                    packet, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    self._cleanup_stale_frames()
                    continue

                if self._parse_udp_packet(packet, addr):
                    log("[INFO] 按下 q / ESC，程序退出")
                    return 0
                self._cleanup_stale_frames()
        except KeyboardInterrupt:
            print()
            log("[INFO] Ctrl+C 退出")
            return 0
        finally:
            self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="杰理 UDP JPEG 流 + RKNN 检测 + LCD 显示（单输出调试版）")
    p.add_argument("--bind-ip", default="0.0.0.0")
    p.add_argument("--port", type=int, default=2224)
    p.add_argument("--device-ip", default="192.168.1.1")
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--cleanup-timeout", type=float, default=3.0)

    p.add_argument("--model", required=True)
    p.add_argument("--labels", default=None)
    p.add_argument("--input-size", nargs=2, type=int, default=[640, 640], metavar=("W", "H"))
    p.add_argument("--obj-thresh", type=float, default=0.25)
    p.add_argument("--nms-thresh", type=float, default=0.45)
    p.add_argument("--max-det", type=int, default=5)
    p.add_argument("--agnostic-nms", action="store_true")
    p.add_argument("--bgr-input", action="store_true")
    p.add_argument("--single-core", action="store_true")

    p.add_argument("--no-window", action="store_true")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--display-size", nargs=2, type=int, default=None, metavar=("W", "H"))
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[FATAL] .rknn 模型不存在: {model_path}")
        return 1

    detector = YoloRknnDetector(
        model_path=str(model_path),
        labels_path=args.labels,
        input_size=(args.input_size[1], args.input_size[0]),
        obj_thresh=args.obj_thresh,
        nms_thresh=args.nms_thresh,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
        use_rgb=not args.bgr_input,
        use_all_cores=not args.single_core,
        verbose=args.verbose,
    )

    display_size = tuple(args.display_size) if args.display_size is not None else None

    app = JieliRknnUdpInfer(
        bind_ip=args.bind_ip,
        bind_port=args.port,
        device_ip=None if args.no_filter else args.device_ip,
        cleanup_timeout=args.cleanup_timeout,
        show_window=not args.no_window,
        fullscreen=args.fullscreen,
        display_size=display_size,
        detector=detector,
        verbose=args.verbose,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
