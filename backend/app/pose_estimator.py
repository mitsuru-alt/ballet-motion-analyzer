from __future__ import annotations

"""
Pirouette Analyzer - MediaPipe PoseLandmarker Task API ラッパー

ピルエット（回転）専門の軽量パイプライン。
1回の動画スキャンで回転検出＋ポーズ評価を同時に行う。
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from pathlib import Path
import os

from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision

# PoseLandmarker モデル設定
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_MODEL_VARIANT = os.environ.get("POSE_MODEL", "lite")
_MODEL_FILENAMES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}
_TASK_MODEL_PATH = _MODEL_DIR / _MODEL_FILENAMES.get(_MODEL_VARIANT, "pose_landmarker_lite.task")
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    f"pose_landmarker/pose_landmarker_{_MODEL_VARIANT}/float16/latest/"
    f"pose_landmarker_{_MODEL_VARIANT}.task"
)

# フレームの最大長辺（メモリ節約）
_MAX_FRAME_DIM = int(os.environ.get("MAX_FRAME_DIM", "480"))


def _downscale_frame(frame: np.ndarray, max_dim: int = _MAX_FRAME_DIM) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _save_video_tempfile(video_bytes: bytes, original_filename: str | None = None) -> str:
    """動画バイトを一時ファイルに保存（iOS .mov対応）"""
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
class FrameResult:
    """1フレームの骨格検出結果"""
    frame_index: int
    timestamp_ms: float
    landmarks: list
    world_landmarks: list
    image_width: int
    image_height: int

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


@dataclass
class SinglePassResult:
    """1回の動画スキャンで得られる全データ"""
    frame_results: list[FrameResult]       # 3fps ポーズ評価用
    dense_frames: list[dict]               # 15fps 回転検出用（肩腰X座標のみ）
    source_fps: float
    video_duration_ms: float
    best_frame_jpg: bytes | None = None    # ベストフレーム画像


class PoseEstimator:
    """ピルエット専門の姿勢推定エンジン"""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        sample_fps: int = 3,
        dense_fps: int = 15,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.sample_fps = sample_fps
        self.dense_fps = dense_fps

    def _ensure_task_model(self) -> str:
        if _TASK_MODEL_PATH.exists():
            return str(_TASK_MODEL_PATH)
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"[PoseEstimator] Downloading model to {_TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(_TASK_MODEL_URL, str(_TASK_MODEL_PATH))
        print("[PoseEstimator] Download complete.")
        return str(_TASK_MODEL_PATH)

    def _create_landmarker(self) -> mp_vision.PoseLandmarker:
        model_path = self._ensure_task_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=model_path),
            num_poses=1,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    # ================================================================
    # 画像処理
    # ================================================================

    def process_image(self, image_bytes: bytes) -> FrameResult | None:
        """静止画から骨格を検出"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None
        image = _downscale_frame(image)
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._create_landmarker() as landmarker:
            result = landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        return FrameResult(
            frame_index=0,
            timestamp_ms=0.0,
            landmarks=list(result.pose_landmarks[0]),
            world_landmarks=list(result.pose_world_landmarks[0]) if result.pose_world_landmarks else [],
            image_width=w,
            image_height=h,
        )

    # ================================================================
    # 動画処理（シングルパス）
    # ================================================================

    def process_video_single_pass(self, video_path: str) -> SinglePassResult:
        """
        1回の動画スキャンでポーズ評価用 + 回転検出用データを同時取得。

        - 3fps: 全ランドマーク記録 → ポーズ評価
        - 15fps: 肩腰X座標のみ記録 → 回転検出
        - ベストフレーム画像も同時抽出
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return SinglePassResult([], [], 30.0, 0.0)

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        video_duration_ms = (total_frames / source_fps) * 1000 if source_fps > 0 else 0.0

        # フレーム間引き間隔
        pose_interval = max(1, int(source_fps / self.sample_fps))     # 3fps用
        dense_interval = max(1, int(source_fps / self.dense_fps))     # 15fps用

        frame_results = []
        dense_frames = []
        frame_index = 0

        with self._create_landmarker() as landmarker:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                is_pose_frame = (frame_index % pose_interval == 0)
                is_dense_frame = (frame_index % dense_interval == 0)

                if is_pose_frame or is_dense_frame:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    small = _downscale_frame(frame)
                    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    detection = landmarker.detect(mp_image)

                    if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                        lms = detection.pose_landmarks[0]

                        # 3fps: フルランドマーク記録
                        if is_pose_frame:
                            h, w = small.shape[:2]
                            wl = list(detection.pose_world_landmarks[0]) if detection.pose_world_landmarks else []
                            frame_results.append(FrameResult(
                                frame_index=frame_index,
                                timestamp_ms=timestamp_ms,
                                landmarks=list(lms),
                                world_landmarks=wl,
                                image_width=w,
                                image_height=h,
                            ))

                        # 15fps: 回転用座標のみ
                        if is_dense_frame:
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

        # ベストフレーム画像は後で抽出（evaluate後にbest_frame_indexが決まる）
        return SinglePassResult(
            frame_results=frame_results,
            dense_frames=dense_frames,
            source_fps=source_fps,
            video_duration_ms=video_duration_ms,
        )

    def extract_frame_image_from_path(
        self, video_path: str, frame_index: int,
    ) -> bytes | None:
        """動画から指定フレームをJPEG画像として抽出"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes()
