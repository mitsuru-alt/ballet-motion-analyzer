from __future__ import annotations

"""
MediaPipe PoseLandmarker Task API ラッパー

動画・静止画から骨格ランドマークを抽出する軽量パイプライン。
すべての検出に PoseLandmarker Task API を使用（Linux headless 環境互換）。
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from pathlib import Path
import os
import math

from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

# PoseLandmarker モデルファイルのパス
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_TASK_MODEL_PATH = _MODEL_DIR / "pose_landmarker_heavy.task"
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)

# フレームの最大長辺ピクセル数（メモリ節約のためリサイズ）
_MAX_FRAME_DIM = int(os.environ.get("MAX_FRAME_DIM", "640"))


def _downscale_frame(frame: np.ndarray, max_dim: int = _MAX_FRAME_DIM) -> np.ndarray:
    """長辺が max_dim を超える場合にアスペクト比を維持して縮小する"""
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _save_video_tempfile(video_bytes: bytes, original_filename: str | None = None) -> str:
    """動画バイトを一時ファイルに保存する。
    iOS の .mov 等にも対応するため、元の拡張子を維持する。"""
    import tempfile
    suffix = ".mp4"
    if original_filename:
        ext = os.path.splitext(original_filename)[1].lower()
        if ext in (".mov", ".mp4", ".avi", ".webm", ".m4v", ".3gp"):
            suffix = ext
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, video_bytes)
    finally:
        os.close(fd)
    return tmp_path


@dataclass
class PersonResult:
    """1人分の骨格検出結果"""
    person_id: int
    landmarks: list
    world_landmarks: list

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "landmarks": [
                {
                    "index": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 0.0,
                }
                for i, lm in enumerate(self.landmarks)
            ],
        }


@dataclass
class MultiFrameResult:
    """マルチパーソン（2人）の1フレーム解析結果"""
    frame_index: int
    timestamp_ms: float
    persons: list[PersonResult]
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "persons": [p.to_dict() for p in self.persons],
        }


@dataclass
class FrameResult:
    """1フレームの解析結果"""
    frame_index: int
    timestamp_ms: float
    landmarks: list
    world_landmarks: list
    image_width: int
    image_height: int

    def landmark_to_pixel(self, idx: int) -> tuple[int, int]:
        lm = self.landmarks[idx]
        return (int(lm.x * self.image_width), int(lm.y * self.image_height))

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "landmarks": [
                {
                    "index": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility if hasattr(lm, 'visibility') else 0.0,
                }
                for i, lm in enumerate(self.landmarks)
            ],
        }


class PoseEstimator:
    """
    MediaPipe PoseLandmarker Task API による姿勢推定エンジン

    すべての検出に Task API を使用。solutions API に依存しないため
    Linux headless 環境（Render.com等）でも確実に動作する。
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        sample_fps: int = 3,
    ):
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.sample_fps = sample_fps

    def _ensure_task_model(self) -> str:
        """PoseLandmarker の .task モデルファイルを確保する。"""
        if _TASK_MODEL_PATH.exists():
            return str(_TASK_MODEL_PATH)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"[PoseEstimator] Downloading PoseLandmarker model to {_TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(_TASK_MODEL_URL, str(_TASK_MODEL_PATH))
        print("[PoseEstimator] Download complete.")
        return str(_TASK_MODEL_PATH)

    def _create_landmarker(self, num_poses: int = 1) -> mp_vision.PoseLandmarker:
        """PoseLandmarker インスタンスを生成"""
        model_path = self._ensure_task_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=model_path
            ),
            num_poses=num_poses,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    def _detect_task_single(
        self, image: np.ndarray, frame_index: int, timestamp_ms: float
    ) -> FrameResult | None:
        """Task API で単一フレームの骨格検出"""
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._create_landmarker(num_poses=1) as landmarker:
            result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        lms = list(result.pose_landmarks[0])
        wl = list(result.pose_world_landmarks[0]) if result.pose_world_landmarks else []

        return FrameResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            landmarks=lms,
            world_landmarks=wl,
            image_width=w,
            image_height=h,
        )

    # ================================================================
    # 画像処理
    # ================================================================

    def process_image(self, image_bytes: bytes) -> FrameResult | None:
        """静止画バイト列から骨格を検出"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = _downscale_frame(image)
        return self._detect_task_single(image, frame_index=0, timestamp_ms=0.0)

    # ================================================================
    # 動画処理（単一人物）
    # ================================================================

    def process_video(
        self, video_bytes: bytes, original_filename: str | None = None
    ) -> list[FrameResult]:
        """動画バイト列からフレームをサンプリングして骨格検出"""
        tmp_path = _save_video_tempfile(video_bytes, original_filename)
        try:
            return self._process_video_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _process_video_file(self, video_path: str) -> list[FrameResult]:
        """動画ファイルパスから処理（Task API使用）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = max(1, int(source_fps / self.sample_fps))

        results = []
        frame_index = 0

        with self._create_landmarker(num_poses=1) as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    small = _downscale_frame(frame)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb
                    )
                    detection = landmarker.detect(mp_image)

                    if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                        h, w = small.shape[:2]
                        lms = list(detection.pose_landmarks[0])
                        wl = list(detection.pose_world_landmarks[0]) if detection.pose_world_landmarks else []
                        results.append(FrameResult(
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            landmarks=lms,
                            world_landmarks=wl,
                            image_width=w,
                            image_height=h,
                        ))

                frame_index += 1

        cap.release()
        return results

    def process_video_dense(
        self, video_bytes: bytes, dense_fps: int = 15,
        original_filename: str | None = None,
    ) -> tuple[list[dict], float]:
        """回転検出用の高頻度サンプリング（Task API使用）"""
        tmp_path = _save_video_tempfile(video_bytes, original_filename)

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return [], 30.0

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = max(1, int(source_fps / dense_fps))

            dense_frames = []
            frame_index = 0

            with self._create_landmarker(num_poses=1) as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % frame_interval == 0:
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        small = _downscale_frame(frame)
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=rgb
                        )
                        detection = landmarker.detect(mp_image)

                        if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                            lms = detection.pose_landmarks[0]
                            dense_frames.append({
                                "frame_index": frame_index,
                                "timestamp_ms": timestamp_ms,
                                "left_shoulder_x": lms[11].x,
                                "right_shoulder_x": lms[12].x,
                                "left_hip_x": lms[23].x,
                                "right_hip_x": lms[24].x,
                            })

                    frame_index += 1

            cap.release()
            return dense_frames, source_fps
        finally:
            os.unlink(tmp_path)

    # ================================================================
    # ユーティリティ
    # ================================================================

    def extract_frame_image(
        self, video_bytes: bytes, frame_index: int,
        original_filename: str | None = None,
    ) -> bytes | None:
        """動画から指定フレームをJPEG画像として抽出"""
        tmp_path = _save_video_tempfile(video_bytes, original_filename)
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return encoded.tobytes()
        finally:
            os.unlink(tmp_path)

    def get_video_duration_ms(
        self, video_bytes: bytes,
        original_filename: str | None = None,
    ) -> float:
        """動画の総時間をミリ秒で返す"""
        tmp_path = _save_video_tempfile(video_bytes, original_filename)
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return 0.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            return (total_frames / fps) * 1000
        finally:
            os.unlink(tmp_path)

    def draw_skeleton(
        self,
        image_bytes: bytes,
        frame_result: FrameResult,
    ) -> bytes:
        """骨格をオーバーレイ描画した画像を返す（フォールバック用）"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Task API のランドマークからOpenCVで直接描画
        h, w = image.shape[:2]
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
            (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
        ]

        for a, b in connections:
            if a < len(frame_result.landmarks) and b < len(frame_result.landmarks):
                la, lb = frame_result.landmarks[a], frame_result.landmarks[b]
                vis_a = la.visibility if hasattr(la, 'visibility') else 0.0
                vis_b = lb.visibility if hasattr(lb, 'visibility') else 0.0
                if vis_a > 0.5 and vis_b > 0.5:
                    pt1 = (int(la.x * w), int(la.y * h))
                    pt2 = (int(lb.x * w), int(lb.y * h))
                    cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        for lm in frame_result.landmarks:
            vis = lm.visibility if hasattr(lm, 'visibility') else 0.0
            if vis > 0.5:
                pt = (int(lm.x * w), int(lm.y * h))
                cv2.circle(image, pt, 4, (0, 0, 255), -1)

        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes()

    # ================================================================
    # マルチパーソン検出（パ・ド・ドゥ対応）
    # ================================================================

    def _assign_person_ids(
        self,
        pose_landmarks_list: list,
        prev_hip_centers: list[tuple[float, float]] | None = None,
    ) -> list[tuple[int, list]]:
        """検出された複数人にPerson IDを割り当てる。"""
        if len(pose_landmarks_list) < 2:
            return [(i, lm) for i, lm in enumerate(pose_landmarks_list)]

        def hip_center(lms):
            lh, rh = lms[23], lms[24]
            return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)

        centers = [hip_center(lms) for lms in pose_landmarks_list]

        if prev_hip_centers is None:
            order = sorted(range(len(centers)), key=lambda i: centers[i][0])
        else:
            order = []
            used = set()
            for prev_c in prev_hip_centers:
                best_idx = -1
                best_dist = float("inf")
                for j in range(len(centers)):
                    if j in used:
                        continue
                    d = math.hypot(centers[j][0] - prev_c[0], centers[j][1] - prev_c[1])
                    if d < best_dist:
                        best_dist = d
                        best_idx = j
                if best_idx >= 0:
                    order.append(best_idx)
                    used.add(best_idx)
            for j in range(len(centers)):
                if j not in used:
                    order.append(j)

        return [(pid, pose_landmarks_list[idx]) for pid, idx in enumerate(order)]

    def process_image_multi(self, image_bytes: bytes) -> MultiFrameResult | None:
        """静止画から2人の骨格を検出する。"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        image = _downscale_frame(image)
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._create_landmarker(num_poses=2) as landmarker:
            result = landmarker.detect(mp_image)

        if len(result.pose_landmarks) < 2:
            return None

        assigned = self._assign_person_ids(
            [list(lms) for lms in result.pose_landmarks]
        )
        world_lms = result.pose_world_landmarks

        persons = []
        for pid, lms in assigned:
            wl = list(world_lms[pid]) if pid < len(world_lms) else []
            persons.append(PersonResult(
                person_id=pid,
                landmarks=lms,
                world_landmarks=wl,
            ))

        return MultiFrameResult(
            frame_index=0,
            timestamp_ms=0.0,
            persons=sorted(persons, key=lambda p: p.person_id),
            image_width=w,
            image_height=h,
        )

    def process_video_multi(
        self, video_bytes: bytes, original_filename: str | None = None
    ) -> list[MultiFrameResult]:
        """動画から2人の骨格を継続トラッキングする。"""
        tmp_path = _save_video_tempfile(video_bytes, original_filename)

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return []

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = max(1, int(source_fps / self.sample_fps))

            results = []
            frame_index = 0
            prev_hip_centers = None

            with self._create_landmarker(num_poses=2) as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % frame_interval == 0:
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        small = _downscale_frame(frame)
                        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=rgb
                        )
                        detection = landmarker.detect(mp_image)

                        if len(detection.pose_landmarks) >= 2:
                            h, w = small.shape[:2]
                            lms_lists = [list(lms) for lms in detection.pose_landmarks]
                            assigned = self._assign_person_ids(
                                lms_lists, prev_hip_centers
                            )
                            world_lms = detection.pose_world_landmarks

                            persons = []
                            for pid, lms in assigned:
                                wl = list(world_lms[pid]) if pid < len(world_lms) else []
                                persons.append(PersonResult(
                                    person_id=pid,
                                    landmarks=lms,
                                    world_landmarks=wl,
                                ))

                            persons.sort(key=lambda p: p.person_id)
                            results.append(MultiFrameResult(
                                frame_index=frame_index,
                                timestamp_ms=timestamp_ms,
                                persons=persons,
                                image_width=w,
                                image_height=h,
                            ))

                            prev_hip_centers = []
                            for p in persons:
                                lh, rh = p.landmarks[23], p.landmarks[24]
                                prev_hip_centers.append(
                                    ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)
                                )

                    frame_index += 1

            cap.release()
            return results
        finally:
            os.unlink(tmp_path)
