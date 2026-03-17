from __future__ import annotations

"""
MediaPipe BlazePose ラッパー

動画・静止画から骨格ランドマークを抽出する軽量パイプライン。
リアルタイム性を重視し、動画は1-5fpsでサンプリングする。
マルチパーソン検出（パ・ド・ドゥ対応）は PoseLandmarker Task API を使用。
"""

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions as mp_solutions
from dataclasses import dataclass, field
from pathlib import Path
import os
import math

from mediapipe.tasks import python as mp_tasks_python
from mediapipe.tasks.python import vision as mp_vision


mp_pose = mp_solutions.pose
mp_drawing = mp_solutions.drawing_utils
mp_drawing_styles = mp_solutions.drawing_styles

# PoseLandmarker モデルファイルのパス
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
_TASK_MODEL_PATH = _MODEL_DIR / "pose_landmarker_heavy.task"
_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)


@dataclass
class PersonResult:
    """1人分の骨格検出結果"""
    person_id: int          # 0 = Dancer A, 1 = Dancer B
    landmarks: list         # 33 NormalizedLandmarks (x,y,z,visibility,presence)
    world_landmarks: list   # 33 WorldLandmarks

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
    persons: list[PersonResult]  # 2人分
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
    landmarks: list          # MediaPipe NormalizedLandmark のリスト
    world_landmarks: list    # 3Dワールド座標のランドマーク
    image_width: int
    image_height: int

    def landmark_to_pixel(self, idx: int) -> tuple[int, int]:
        """ランドマークインデックスをピクセル座標に変換"""
        lm = self.landmarks[idx]
        return (int(lm.x * self.image_width), int(lm.y * self.image_height))

    def to_dict(self) -> dict:
        """JSON変換用の辞書を返す"""
        return {
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "landmarks": [
                {
                    "index": i,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                }
                for i, lm in enumerate(self.landmarks)
            ],
        }


class PoseEstimator:
    """
    MediaPipe BlazePose による姿勢推定エンジン

    Parameters:
        model_complexity: 0(Lite), 1(Full), 2(Heavy) — 精度と速度のトレードオフ
        min_detection_confidence: 検出閾値 (0.0-1.0)
        min_tracking_confidence: トラッキング閾値 (動画用)
        sample_fps: 動画処理時のサンプリングFPS (軽量化)
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

    def process_image(self, image_bytes: bytes) -> FrameResult | None:
        """
        静止画バイト列から骨格を検出

        Args:
            image_bytes: 画像ファイルのバイト列 (JPEG/PNG)

        Returns:
            FrameResult or None (検出失敗時)
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        return self._detect_single(image, frame_index=0, timestamp_ms=0.0)

    def process_video(self, video_bytes: bytes) -> list[FrameResult]:
        """
        動画バイト列からフレームをサンプリングして骨格検出

        sample_fps に基づいてフレームを間引き、軽量に処理する。

        Args:
            video_bytes: 動画ファイルのバイト列

        Returns:
            検出成功フレームの FrameResult リスト
        """
        import tempfile
        import os

        # OpenCV は直接バイト列から動画を読めないため一時ファイル経由
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            return self._process_video_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _process_video_file(self, video_path: str) -> list[FrameResult]:
        """動画ファイルパスから処理"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # sample_fps に基づくフレーム間引き間隔
        frame_interval = max(1, int(source_fps / self.sample_fps))

        results = []
        frame_index = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        ) as pose:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % frame_interval == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb)

                    if pose_results.pose_landmarks:
                        h, w = frame.shape[:2]
                        results.append(FrameResult(
                            frame_index=frame_index,
                            timestamp_ms=timestamp_ms,
                            landmarks=list(pose_results.pose_landmarks.landmark),
                            world_landmarks=list(
                                pose_results.pose_world_landmarks.landmark
                            ) if pose_results.pose_world_landmarks else [],
                            image_width=w,
                            image_height=h,
                        ))

                frame_index += 1

        cap.release()
        return results

    def process_video_dense(self, video_bytes: bytes, dense_fps: int = 15) -> tuple[list[dict], float]:
        """
        回転検出用の高頻度サンプリング

        通常の3fpsでは回転を検出できないため、15fpsでサンプリングし
        肩・腰のX座標のみを記録する軽量パス。

        Args:
            video_bytes: 動画ファイルのバイト列
            dense_fps: サンプリングFPS (デフォルト15)

        Returns:
            (dense_frames, source_fps) のタプル
            dense_frames: [{"frame_index": int, "timestamp_ms": float,
                           "left_shoulder_x": float, "right_shoulder_x": float,
                           "left_hip_x": float, "right_hip_x": float}]
            source_fps: 元動画のFPS
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return [], 30.0

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = max(1, int(source_fps / dense_fps))

            dense_frames = []
            frame_index = 0

            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as pose:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % frame_interval == 0:
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(rgb)

                        if pose_results.pose_landmarks:
                            lms = pose_results.pose_landmarks.landmark
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

    def extract_frame_image(self, video_bytes: bytes, frame_index: int) -> bytes | None:
        """動画から指定フレームをJPEG画像として抽出"""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

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

    def get_video_duration_ms(self, video_bytes: bytes) -> float:
        """動画の総時間をミリ秒で返す"""
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

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

    def _detect_single(
        self, image: np.ndarray, frame_index: int, timestamp_ms: float
    ) -> FrameResult | None:
        """単一フレームの骨格検出"""
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            min_detection_confidence=self.min_detection_confidence,
        ) as pose:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                return None

            h, w = image.shape[:2]
            return FrameResult(
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                landmarks=list(results.pose_landmarks.landmark),
                world_landmarks=list(
                    results.pose_world_landmarks.landmark
                ) if results.pose_world_landmarks else [],
                image_width=w,
                image_height=h,
            )

    def draw_skeleton(
        self,
        image_bytes: bytes,
        frame_result: FrameResult,
    ) -> bytes:
        """
        骨格をオーバーレイ描画した画像を返す (サーバーサイドレンダリング用)

        Args:
            image_bytes: 元画像のバイト列
            frame_result: 検出結果

        Returns:
            骨格描画済み画像のJPEGバイト列
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # MediaPipe の描画ユーティリティ用にランドマークオブジェクトを再構築
        landmark_proto = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for lm in frame_result.landmarks:
            landmark_proto.landmark.append(
                mp.framework.formats.landmark_pb2.NormalizedLandmark(
                    x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility,
                )
            )

        mp_drawing.draw_landmarks(
            image,
            landmark_proto,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        _, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return encoded.tobytes()

    # ================================================================
    # マルチパーソン検出（パ・ド・ドゥ対応）
    # ================================================================

    def _ensure_task_model(self) -> str:
        """PoseLandmarker の .task モデルファイルを確保する。
        存在しなければダウンロードしてキャッシュする。"""
        if _TASK_MODEL_PATH.exists():
            return str(_TASK_MODEL_PATH)

        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"[PoseEstimator] Downloading PoseLandmarker model to {_TASK_MODEL_PATH}...")
        urllib.request.urlretrieve(_TASK_MODEL_URL, str(_TASK_MODEL_PATH))
        print("[PoseEstimator] Download complete.")
        return str(_TASK_MODEL_PATH)

    def _assign_person_ids(
        self,
        pose_landmarks_list: list,
        prev_hip_centers: list[tuple[float, float]] | None = None,
    ) -> list[tuple[int, list]]:
        """検出された複数人にPerson IDを割り当てる。

        初回: X座標が小さい方が person_id=0
        以降: 前フレームのhip中心との最近傍マッチング
        """
        if len(pose_landmarks_list) < 2:
            return [(i, lm) for i, lm in enumerate(pose_landmarks_list)]

        def hip_center(lms):
            lh, rh = lms[23], lms[24]
            return ((lh.x + rh.x) / 2.0, (lh.y + rh.y) / 2.0)

        centers = [hip_center(lms) for lms in pose_landmarks_list]

        if prev_hip_centers is None:
            # 初回: X座標でソート（左=0, 右=1）
            order = sorted(range(len(centers)), key=lambda i: centers[i][0])
        else:
            # 最近傍マッチング
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
            # 残りの人を追加
            for j in range(len(centers)):
                if j not in used:
                    order.append(j)

        return [(pid, pose_landmarks_list[idx]) for pid, idx in enumerate(order)]

    def process_image_multi(self, image_bytes: bytes) -> MultiFrameResult | None:
        """静止画から2人の骨格を検出する。

        Returns:
            MultiFrameResult (2人検出時) or None (2人未満)
        """
        model_path = self._ensure_task_model()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return None

        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks_python.BaseOptions(
                model_asset_path=model_path
            ),
            num_poses=2,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
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

    def process_video_multi(self, video_bytes: bytes) -> list[MultiFrameResult]:
        """動画から2人の骨格を継続トラッキングする。

        Returns:
            2人検出に成功したフレームの MultiFrameResult リスト
        """
        import tempfile

        model_path = self._ensure_task_model()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                return []

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_interval = max(1, int(source_fps / self.sample_fps))

            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_tasks_python.BaseOptions(
                    model_asset_path=model_path
                ),
                num_poses=2,
                min_pose_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )

            results = []
            frame_index = 0
            prev_hip_centers = None

            with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_index % frame_interval == 0:
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(
                            image_format=mp.ImageFormat.SRGB, data=rgb
                        )
                        detection = landmarker.detect(mp_image)

                        if len(detection.pose_landmarks) >= 2:
                            h, w = frame.shape[:2]
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

                            # 次フレーム用にhip中心を記録
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
