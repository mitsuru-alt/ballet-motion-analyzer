from __future__ import annotations

"""
Pirouette Analyzer - MediaPipe PoseLandmarker Task API ラッパー

ピルエット（回転）専門の軽量パイプライン。
1回の動画スキャンで回転検出＋ポーズ評価を同時に行う。

高速化:
- Landmarkerをキャッシュ（毎回モデルロードしない）
- 不要フレームはgrab()でスキップ（デコードしない）
- dense_fps=10に削減（15→10）
- ベストフレームをスキャン中に保存（動画再オープン不要）
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from pathlib import Path
import os
import threading

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
    frame_results: list[FrameResult]       # ポーズ評価用
    dense_frames: list[dict]               # 回転検出用（肩腰X座標のみ）
    source_fps: float
    video_duration_ms: float
    best_frame_jpg: bytes | None = None    # ベストフレーム画像
    frame_images: dict[int, np.ndarray] | None = None  # フレーム番号→画像


def _ensure_task_model() -> str:
    """モデルファイルが無ければダウンロード"""
    if _TASK_MODEL_PATH.exists():
        return str(_TASK_MODEL_PATH)
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    print(f"[PoseEstimator] Downloading model to {_TASK_MODEL_PATH}...")
    urllib.request.urlretrieve(_TASK_MODEL_URL, str(_TASK_MODEL_PATH))
    print("[PoseEstimator] Download complete.")
    return str(_TASK_MODEL_PATH)


# ============================================================
# キャッシュ済みLandmarker（モデルを毎回ロードしない）
# ============================================================
_landmarker_cache: mp_vision.PoseLandmarker | None = None
_landmarker_lock = threading.Lock()


def _get_landmarker(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> mp_vision.PoseLandmarker:
    """Landmarkerをキャッシュして返す（初回のみモデルロード）"""
    global _landmarker_cache
    if _landmarker_cache is not None:
        return _landmarker_cache
    with _landmarker_lock:
        if _landmarker_cache is not None:
            return _landmarker_cache
        model_path = _ensure_task_model()
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(model_asset_path=model_path),
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        _landmarker_cache = mp_vision.PoseLandmarker.create_from_options(options)
        print("[PoseEstimator] Landmarker cached.")
        return _landmarker_cache


class PoseEstimator:
    """ピルエット専門の姿勢推定エンジン"""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        sample_fps: int = 2,       # 3→2に削減（十分な精度）
        dense_fps: int = 10,       # 15→10に削減（回転検出には十分）
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.sample_fps = sample_fps
        self.dense_fps = dense_fps

    def _get_landmarker(self) -> mp_vision.PoseLandmarker:
        return _get_landmarker(
            self.min_detection_confidence,
            self.min_tracking_confidence,
        )

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

        landmarker = self._get_landmarker()
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
    # 動画処理（シングルパス・高速版）
    # ================================================================

    def process_video_single_pass(self, video_path: str) -> SinglePassResult:
        """
        1回の動画スキャンでポーズ評価用 + 回転検出用データを同時取得。

        高速化:
        - 不要フレームはgrab()でスキップ（デコードしない）
        - Landmarkerはキャッシュ済み
        - ポーズ用フレームの画像もメモリに保持（後で再オープン不要）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return SinglePassResult([], [], 30.0, 0.0)

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # iOS .mov では FRAME_COUNT が 0 を返すことがある
        has_total = total_frames_raw > 0
        video_duration_ms = (total_frames_raw / source_fps) * 1000 if has_total and source_fps > 0 else 0.0

        # フレーム間引き間隔
        pose_interval = max(1, int(source_fps / self.sample_fps))     # ポーズ評価用
        dense_interval = max(1, int(source_fps / self.dense_fps))     # 回転検出用

        frame_results = []
        dense_frames = []
        frame_images: dict[int, np.ndarray] = {}  # ベストフレーム抽出用

        landmarker = self._get_landmarker()
        frame_index = 0

        while True:
            is_pose_frame = (frame_index % pose_interval == 0)
            is_dense_frame = (frame_index % dense_interval == 0)
            need_this_frame = is_pose_frame or is_dense_frame

            if not need_this_frame:
                # 不要フレーム: grab()でスキップ（デコードしない = 高速）
                if not cap.grab():
                    break
                frame_index += 1
                continue

            ret, frame = cap.read()
            if not ret:
                break

            timestamp_ms = (frame_index / source_fps) * 1000
            small = _downscale_frame(frame)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = landmarker.detect(mp_image)

            if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                lms = detection.pose_landmarks[0]

                # ポーズ評価用: フルランドマーク記録
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
                    # 元フレーム（縮小済み）を保持（メモリ節約）
                    frame_images[frame_index] = small.copy()

                # 回転検出用: 座標のみ
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

        # total_frames が取れなかった場合、実際に読めたフレーム数から計算
        if not has_total and frame_index > 0:
            video_duration_ms = (frame_index / source_fps) * 1000

        return SinglePassResult(
            frame_results=frame_results,
            dense_frames=dense_frames,
            source_fps=source_fps,
            video_duration_ms=video_duration_ms,
            frame_images=frame_images,
        )

    def extract_best_frame_image(
        self,
        scan_result: SinglePassResult,
        best_frame_index: int,
        video_path: str | None = None,
    ) -> bytes | None:
        """ベストフレームをJPEG画像として取得（スキャン時の画像を使用）"""
        # まずスキャン中に保持した画像を使う（高速）
        if scan_result.frame_images and best_frame_index in scan_result.frame_images:
            frame = scan_result.frame_images[best_frame_index]
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return encoded.tobytes()

        # フォールバック: 動画から直接読み込み
        if video_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_index)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return encoded.tobytes()

        return None
