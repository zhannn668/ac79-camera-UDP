#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 RK3588 上接收杰理设备发来的 UDP JPEG 实时流，
使用 RKNNLite 进行 YOLOv8 目标检测，并把结果显示到 LCD 屏幕。

这版修正了两个关键问题：
1. 输入统一补 batch 维，按 4 维 NHWC 喂给 RKNNLite；
2. 同时兼容两类常见输出：
   - 6 个输出张量（Rockchip Model Zoo 常见 YOLOv8 后处理前输出）
   - 1 个输出张量（Ultralytics 直接导出的 ONNX/RKNN 常见输出）

适用前提：
- 杰理端 OPEN_RT_STREAM 使用 JPEG 模式（format=0）
- 当前 .rknn 是目标检测模型
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
H264_TYPE_VIDEO = 0x03
PREVIEW_TYPE = 0x04
DATE_TIME_TYPE = 0x05
MEDIA_INFO_TYPE = 0x06
PLAY_OVER_TYPE = 0x07
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
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.last_seen = False

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
        if is_last:
            self.last_seen = True

    def is_complete(self) -> bool:
        return self.received >= self.frame_size

    def age(self) -> float:
        return time.time() - self.updated_at

    def to_bytes(self) -> bytes:
        return bytes(self.buf[: self.frame_size])


class YoloV8RknnDetector:
    def __init__(
        self,
        model_path: str,
        labels_path: Optional[str],
        input_size: Tuple[int, int],
        obj_thresh: float,
        nms_thresh: float,
        use_rgb: bool,
        use_all_cores: bool,
        verbose: bool = False,
    ) -> None:
        self.model_path = model_path
        self.labels = self._load_labels(labels_path)
        self.input_h, self.input_w = input_size
        self.obj_thresh = obj_thresh
        self.nms_thresh = nms_thresh
        self.use_rgb = use_rgb
        self.verbose = verbose
        self.rknn = RKNNLite()
        self._logged_output_shapes = False

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
            log("[WARN] 未提供 labels.txt，将使用 class_0/class_1/... 占位名")
            return []
        path = Path(labels_path)
        if not path.exists():
            log(f"[WARN] labels 文件不存在: {labels_path}，将使用占位名")
            return []
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return lines

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

        info = LetterBoxInfo(
            ratio=ratio,
            pad_w=pad_w,
            pad_h=pad_h,
            new_w=new_w,
            new_h=new_h,
        )
        return canvas, info

    @staticmethod
    def _reshape_reg(reg: np.ndarray) -> np.ndarray:
        arr = np.asarray(reg)
        arr = np.squeeze(arr)

        if arr.ndim != 3:
            raise ValueError(f"reg 输出维度异常: shape={reg.shape}")

        if arr.shape[0] == 4 * DFL_LEN:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] == 4 * DFL_LEN:
            pass
        else:
            raise ValueError(f"无法识别 reg 输出布局: shape={reg.shape}")

        h, w, c = arr.shape
        return arr.reshape(h, w, 4, DFL_LEN)

    @staticmethod
    def _reshape_cls(cls: np.ndarray) -> np.ndarray:
        arr = np.asarray(cls)
        arr = np.squeeze(arr)

        if arr.ndim != 3:
            raise ValueError(f"cls 输出维度异常: shape={cls.shape}")

        if arr.shape[0] < 200 and arr.shape[1] > 4 and arr.shape[2] > 4:
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.shape[-1] < 200:
            pass
        else:
            raise ValueError(f"无法识别 cls 输出布局: shape={cls.shape}")

        return arr

    @staticmethod
    def _softmax_last_dim(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x, axis=-1, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    def _decode_single_scale(
        self,
        reg: np.ndarray,
        cls: np.ndarray,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        reg_hw = self._reshape_reg(reg)
        cls_hw = self._reshape_cls(cls)
        h, w = cls_hw.shape[:2]

        prob = self._softmax_last_dim(reg_hw)
        project = np.arange(DFL_LEN, dtype=np.float32)
        dist = np.sum(prob * project, axis=-1)

        scores_all = sigmoid(cls_hw)
        class_ids = np.argmax(scores_all, axis=-1)
        scores = np.max(scores_all, axis=-1)

        mask = scores > self.obj_thresh
        if not np.any(mask):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
            )

        ys, xs = np.where(mask)
        chosen_scores = scores[ys, xs].astype(np.float32)
        chosen_classes = class_ids[ys, xs].astype(np.int32)
        chosen_dist = dist[ys, xs]

        cx = (xs.astype(np.float32) + 0.5) * stride
        cy = (ys.astype(np.float32) + 0.5) * stride

        l = chosen_dist[:, 0] * stride
        t = chosen_dist[:, 1] * stride
        r = chosen_dist[:, 2] * stride
        b = chosen_dist[:, 3] * stride

        x1 = cx - l
        y1 = cy - t
        x2 = cx + r
        y2 = cy + b

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        return boxes, chosen_scores, chosen_classes

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
        order = scores.argsort()[::-1]

        keep: List[int] = []
        while order.size > 0:
            i = order[0]
            keep.append(int(i))
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

    def _postprocess_six_outputs(
        self,
        outputs: Sequence[np.ndarray],
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        regs: List[np.ndarray] = []
        clss: List[np.ndarray] = []
        for out in outputs:
            shape = list(np.asarray(out).shape)
            if 64 in shape:
                regs.append(out)
            else:
                clss.append(out)

        if len(regs) != 3 or len(clss) != 3:
            raise ValueError(f"无法把 6 个输出自动分成 3 个 reg + 3 个 cls。reg={len(regs)}, cls={len(clss)}")

        def grid_hw(x: np.ndarray) -> int:
            arr = np.squeeze(np.asarray(x))
            if arr.ndim != 3:
                return -1
            vals = [v for v in arr.shape if v not in (64,) and v < 200]
            vals = [v for v in vals if v > 4]
            return max(vals) if vals else -1

        regs.sort(key=grid_hw, reverse=True)
        clss.sort(key=grid_hw, reverse=True)

        all_boxes: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []
        all_classes: List[np.ndarray] = []

        for reg, cls, stride in zip(regs, clss, STRIDES):
            boxes, scores, class_ids = self._decode_single_scale(reg, cls, stride)
            if len(boxes) == 0:
                continue
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(class_ids)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_classes, axis=0)

        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] -= lb.pad_w
        boxes[:, [1, 3]] -= lb.pad_h
        boxes[:, :4] /= lb.ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = self._nms(cls_boxes, cls_scores, self.nms_thresh)
            for k in keep:
                x1, y1, x2, y2 = cls_boxes[k]
                results.append((int(cls_id), float(cls_scores[k]), (int(x1), int(y1), int(x2), int(y2))))
        return results

    def _postprocess_single_output(
        self,
        output: np.ndarray,
        lb: LetterBoxInfo,
        orig_shape: Tuple[int, int],
    ) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        """
        兼容 Ultralytics 常见单输出：
        - [1, C, N]
        - [1, N, C]
        其中 C 常见为 4+num_classes
        """
        arr = np.asarray(output)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            raise ValueError(f"单输出模型当前只支持 2 维主体张量，实际 shape={output.shape}")

        num_classes = max(1, len(self.labels))
        candidate_channels = 4 + num_classes

        # [C, N] -> [N, C]
        if arr.shape[0] == candidate_channels and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        # [N, C]
        elif arr.shape[1] == candidate_channels:
            pass
        else:
            # 退一步：如果最后一维 >= 5，就当 [N,C] 处理
            if arr.shape[1] >= 5:
                pass
            elif arr.shape[0] >= 5:
                arr = arr.T
            else:
                raise ValueError(f"无法识别单输出张量布局: shape={output.shape}")

        if arr.shape[1] < 5:
            raise ValueError(f"单输出张量列数异常，至少应包含 4 个框参数 + 1 个分数，实际 shape={arr.shape}")

        boxes_xywh = arr[:, :4].astype(np.float32)

        if arr.shape[1] == 5:
            scores = arr[:, 4].astype(np.float32)
            class_ids = np.zeros((arr.shape[0],), dtype=np.int32)
        else:
            cls_scores = arr[:, 4:].astype(np.float32)
            # 这里大多数 Ultralytics 导出已经做过 sigmoid；若没做，数值通常会很奇怪。
            class_ids = np.argmax(cls_scores, axis=1).astype(np.int32)
            scores = np.max(cls_scores, axis=1).astype(np.float32)

        mask = scores > self.obj_thresh
        if not np.any(mask):
            return []

        boxes_xywh = boxes_xywh[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        # 这里按 Ultralytics 常见单输出解释：xywh, 基于 letterbox 后输入图尺度
        cx = boxes_xywh[:, 0]
        cy = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

        boxes = np.stack([x1, y1, x2, y2], axis=1)

        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] -= lb.pad_w
        boxes[:, [1, 3]] -= lb.pad_h
        boxes[:, :4] /= lb.ratio

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h - 1)

        results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
        for cls_id in np.unique(class_ids):
            inds = np.where(class_ids == cls_id)[0]
            cls_boxes = boxes[inds]
            cls_scores = scores[inds]
            keep = self._nms(cls_boxes, cls_scores, self.nms_thresh)
            for k in keep:
                x1, y1, x2, y2 = cls_boxes[k]
                results.append((int(cls_id), float(cls_scores[k]), (int(x1), int(y1), int(x2), int(y2))))
        return results

    def infer(self, image_bgr: np.ndarray) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
        blob, lb = self._letterbox(image_bgr)

        # 当前运行时提示实际需要 NHWC，所以直接按 NHWC 喂 4 维输入
        blob = np.expand_dims(blob, axis=0)
        blob = np.ascontiguousarray(blob)

        outputs = self.rknn.inference(inputs=[blob])
        if outputs is None:
            raise RuntimeError("rknn.inference() 返回了 None")

        if not self._logged_output_shapes:
            shapes = [tuple(np.asarray(o).shape) for o in outputs]
            log(f"[INFO] 模型输出数量: {len(outputs)}")
            log(f"[INFO] 模型输出 shape: {shapes}")
            self._logged_output_shapes = True

        if len(outputs) == 1:
            return self._postprocess_single_output(outputs[0], lb, image_bgr.shape[:2])
        if len(outputs) == 6:
            return self._postprocess_six_outputs(outputs, lb, image_bgr.shape[:2])

        raise ValueError(
            f"当前代码只兼容 1 输出或 6 输出模型，你这个模型实际输出 {len(outputs)} 个。"
        )

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
        detector: YoloV8RknnDetector,
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
        self.packet_count = 0
        self.frame_count = 0
        self.fps_counter = FPSCounter()
        self.window_name = "Jieli RKNN Infer"
        self.window_inited = False

        log(f"[INFO] UDP 监听: {self.bind_ip}:{self.bind_port}")
        if self.device_ip:
            log(f"[INFO] 仅接收设备 IP: {self.device_ip}")
        else:
            log("[INFO] 已关闭设备 IP 过滤")
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
            st = self.frames.pop(seq)
            if self.verbose:
                log(f"[CLEAN] 丢弃超时帧 seq={seq} recv={st.received}/{st.frame_size} age={st.age():.2f}s")

    def _handle_complete_frame(self, state: FrameState) -> bool:
        self.frame_count += 1
        payload = state.to_bytes()

        if not payload.startswith(b"\xFF\xD8"):
            if self.verbose:
                log(f"[DROP] seq={state.seq} 不是 JPEG SOI")
            return False
        if b"\xFF\xD9" not in payload:
            if self.verbose:
                log(f"[DROP] seq={state.seq} 缺少 JPEG EOI")
            return False

        arr = np.frombuffer(payload, dtype=np.uint8)
        frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            if self.verbose:
                log(f"[DROP] seq={state.seq} JPEG 解码失败")
            return False

        try:
            results = self.detector.infer(frame_bgr)
        except Exception as e:
            log(f"[ERR] 推理失败: {e}")
            return False

        vis = self.detector.draw(frame_bgr, results)
        fps = self.fps_counter.update()

        text1 = f"FPS: {fps:.1f}  SEQ: {state.seq}"
        text2 = f"DET: {len(results)}  TS: {state.timestamp}"
        cv2.putText(vis, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(vis, text2, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if self.display_size is not None:
            vis = cv2.resize(vis, self.display_size)

        if self.show_window:
            self._init_window()
            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                return True

        if self.verbose:
            log(f"[FRAME] seq={state.seq} det={len(results)} fps={fps:.1f}")
        return False

    def _parse_udp_packet(self, packet: bytes, addr: Tuple[str, int]) -> bool:
        self.packet_count += 1
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
            if payload_len == 0:
                continue
            if pos + payload_len > plen:
                if self.verbose:
                    log(f"[WARN] 畸形 UDP 包 from={addr[0]}:{addr[1]} payload_len={payload_len}")
                break

            payload = packet[pos: pos + payload_len]
            pos += payload_len

            base_type = media_type & 0x7F
            is_last = bool(media_type & LAST_VIDEO_MARKER)

            if base_type != JPEG_TYPE_VIDEO:
                continue

            st = self.frames.get(seq)
            if st is None:
                st = FrameState(seq=seq, frame_size=frame_size, timestamp=timestamp, media_type=media_type)
                self.frames[seq] = st

            st.add_chunk(offset=offset, payload=payload, is_last=is_last)
            if st.is_complete():
                self.frames.pop(seq, None)
                should_quit = self._handle_complete_frame(st)
                if should_quit:
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

                should_quit = self._parse_udp_packet(packet, addr)
                self._cleanup_stale_frames()
                if should_quit:
                    log("[INFO] 按下 q / ESC，程序退出")
                    return 0
        except KeyboardInterrupt:
            print()
            log("[INFO] Ctrl+C 退出")
            return 0
        finally:
            self.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="杰理 UDP JPEG 流 + RKNN YOLOv8 推理 + LCD 显示")
    p.add_argument("--bind-ip", default="0.0.0.0", help="本地监听 IP，默认 0.0.0.0")
    p.add_argument("--port", type=int, default=2224, help="UDP 端口，默认 2224")
    p.add_argument("--device-ip", default="192.168.1.1", help="只接收该设备 IP，默认 192.168.1.1")
    p.add_argument("--no-filter", action="store_true", help="关闭设备 IP 过滤")
    p.add_argument("--cleanup-timeout", type=float, default=3.0, help="分片帧超时清理时间，默认 3 秒")

    p.add_argument("--model", required=True, help=".rknn 模型路径")
    p.add_argument("--labels", default=None, help="类别文件，每行一个类别名")
    p.add_argument("--input-size", nargs=2, type=int, default=[640, 640], metavar=("W", "H"), help="模型输入尺寸")
    p.add_argument("--obj-thresh", type=float, default=0.25, help="目标置信度阈值")
    p.add_argument("--nms-thresh", type=float, default=0.45, help="NMS 阈值")
    p.add_argument("--bgr-input", action="store_true", help="如果模型需要 BGR 输入而不是 RGB，就加这个参数")
    p.add_argument("--single-core", action="store_true", help="只用默认单核运行，不启用 0_1_2 三核")

    p.add_argument("--no-window", action="store_true", help="不显示窗口，只跑接收和推理")
    p.add_argument("--fullscreen", action="store_true", help="全屏显示到 LCD")
    p.add_argument("--display-size", nargs=2, type=int, default=None, metavar=("W", "H"), help="显示缩放尺寸")
    p.add_argument("--verbose", action="store_true", help="打印更多日志")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[FATAL] .rknn 模型不存在: {model_path}")
        return 1

    detector = YoloV8RknnDetector(
        model_path=str(model_path),
        labels_path=args.labels,
        input_size=(args.input_size[1], args.input_size[0]),
        obj_thresh=args.obj_thresh,
        nms_thresh=args.nms_thresh,
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
