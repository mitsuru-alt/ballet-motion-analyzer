from __future__ import annotations

"""
Ballet Metrics - アラベスク解析のための数学的アルゴリズム

=== アルゴリズム解説 ===

■ 基本原理: ベクトル演算による角度算出

バレエのポーズは「関節間のベクトル」と「そのなす角度」で数値化できる。
MediaPipe BlazePoseは33個のランドマーク(関節点)を返し、
各ランドマークは (x, y, z) の3D座標を持つ。

■ アラベスクの評価に必要な3つの指標:

1. 脚の挙上角度 (Leg Elevation Angle)
   - 軸脚の腰(HIP)から挙脚の足首(ANKLE)へのベクトルと、
     鉛直下向きベクトル(重力方向)のなす角度
   - プロ基準: 90度以上(地面と平行以上)

   計算:
     vec_leg = ankle_raised - hip_raised  (挙脚ベクトル)
     vec_down = [0, 1, 0]                 (鉛直下向き: MediaPipeのy軸は下向き)
     angle = arccos( dot(vec_leg, vec_down) / (|vec_leg| * |vec_down|) )

2. 背中のライン角度 (Back Line Angle)
   - 腰(HIP)から肩(SHOULDER)へのベクトルが
     水平面に対してどれだけ傾いているか
   - プロ基準: 上体は前傾しつつも、肩-腰-挙脚が一直線に近い

   計算:
     vec_torso = shoulder - hip_standing   (体幹ベクトル)
     vec_horizontal = [1, 0, 0]            (水平方向)
     tilt_angle = arccos( dot(vec_torso, vec_horizontal) / ... )

3. 膝の伸展度 (Knee Extension)
   - 挙脚側の 腰→膝ベクトル と 膝→足首ベクトル のなす角度
   - プロ基準: 170-180度 (完全に伸びている)

   計算:
     vec_thigh = knee - hip       (大腿ベクトル)
     vec_shin  = ankle - knee     (下腿ベクトル)
     knee_angle = arccos( dot(vec_thigh, vec_shin) / (|vec_thigh| * |vec_shin|) )
     ※ 180度に近いほど膝が伸びている

■ 3点間の角度の一般公式:

  点A, B(頂点), C があるとき:
    BA = A - B
    BC = C - B
    angle = arccos( dot(BA, BC) / (|BA| * |BC|) )

  これが全ての角度計算の基礎となる。

■ 重心安定性の評価:

  - フレーム間での腰中心点の移動量(分散)を計測
  - 分散が小さいほど重心が安定している
  - center = (left_hip + right_hip) / 2 の時系列データの標準偏差を算出
"""

import math
import numpy as np
from dataclasses import dataclass
from enum import IntEnum


class PoseLandmark(IntEnum):
    """MediaPipe BlazePose の主要ランドマークインデックス"""
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


@dataclass
class ArabesqueMetrics:
    """アラベスク解析結果"""
    leg_elevation_angle: float      # 脚の挙上角度 (度)
    back_line_angle: float          # 背中のライン角度 (度)
    knee_extension_angle: float     # 膝の伸展角度 (度)
    shoulder_hip_leg_alignment: float  # 肩-腰-脚のアライメント (度)
    raised_side: str                # 挙脚側 ("left" or "right")

    @property
    def scores(self) -> dict:
        """各指標を0-100のスコアに変換（マイルド版）"""
        return {
            "leg_elevation": self._score_elevation(),
            "back_line": self._score_back_line(),
            "knee_extension": self._score_knee(),
            "alignment": self._score_alignment(),
        }

    def _score_elevation(self) -> float:
        # 90度以上で満点、30度で20点（マイルド化: 低くても基礎点あり）
        if self.leg_elevation_angle >= 90:
            return 100
        if self.leg_elevation_angle <= 30:
            return 20
        return 20 + (self.leg_elevation_angle - 30) / 60 * 80

    def _score_back_line(self) -> float:
        # 理想は前傾10-35度（幅を広げ）→ その範囲で高得点
        if 10 <= self.back_line_angle <= 35:
            return 100
        deviation = min(abs(self.back_line_angle - 10), abs(self.back_line_angle - 35))
        return max(25, 100 - deviation * 2)

    def _score_knee(self) -> float:
        # 170度以上で満点、130度で30点（マイルド化）
        if self.knee_extension_angle >= 170:
            return 100
        if self.knee_extension_angle <= 130:
            return 30
        return 30 + (self.knee_extension_angle - 130) / 40 * 70

    def _score_alignment(self) -> float:
        # 180度(一直線)で満点、偏差に対して緩やかに減点
        deviation = abs(180 - self.shoulder_hip_leg_alignment)
        return max(25, 100 - deviation * 1.5)


def compute_angle_3points(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> float:
    """
    3点間の角度を計算する (頂点はb)

    Args:
        a: 点A (np.array [x, y, z])
        b: 点B (頂点, np.array [x, y, z])
        c: 点C (np.array [x, y, z])

    Returns:
        角度 (度数法)

    数学的背景:
        BA = A - B, BC = C - B
        cos(θ) = (BA · BC) / (|BA| × |BC|)
        θ = arccos(cos(θ))
    """
    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return 0.0

    cos_angle = np.clip(dot_product / (magnitude_ba * magnitude_bc), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    return math.degrees(angle_rad)


def compute_angle_with_vertical(
    origin: np.ndarray, point: np.ndarray
) -> float:
    """
    ベクトルと鉛直方向のなす角度を計算

    MediaPipeの座標系ではy軸が下向きなので、
    鉛直下向きベクトル = [0, 1, 0]

    Args:
        origin: 始点 (例: 腰)
        point: 終点 (例: 足首)

    Returns:
        鉛直方向からの角度 (度数法)
        0度 = 真下, 90度 = 水平, 180度 = 真上
    """
    vec = point - origin
    vertical = np.array([0.0, 1.0, 0.0])

    dot_product = np.dot(vec, vertical)
    magnitude = np.linalg.norm(vec)

    if magnitude == 0:
        return 0.0

    cos_angle = np.clip(dot_product / magnitude, -1.0, 1.0)
    return math.degrees(np.arccos(cos_angle))


def detect_raised_leg(landmarks: list) -> str:
    """
    どちらの脚が上がっているかを自動検出

    MediaPipeのy座標は下向きが正なので、
    y座標が小さい方の足首が上がっている
    """
    left_ankle_y = landmarks[PoseLandmark.LEFT_ANKLE].y
    right_ankle_y = landmarks[PoseLandmark.RIGHT_ANKLE].y

    return "left" if left_ankle_y < right_ankle_y else "right"


def get_landmark_array(landmark) -> np.ndarray:
    """MediaPipeランドマークをnumpy配列に変換"""
    return np.array([landmark.x, landmark.y, landmark.z])


def analyze_arabesque(landmarks: list) -> ArabesqueMetrics:
    """
    アラベスクのポーズを総合解析する

    Args:
        landmarks: MediaPipe pose_landmarks.landmark のリスト

    Returns:
        ArabesqueMetrics: 各指標の計測値
    """
    raised_side = detect_raised_leg(landmarks)

    # ランドマーク取得（挙脚側 / 軸脚側）
    if raised_side == "left":
        hip_raised = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        knee_raised = get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE])
        ankle_raised = get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE])
        hip_standing = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        shoulder_same = get_landmark_array(landmarks[PoseLandmark.LEFT_SHOULDER])
    else:
        hip_raised = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        knee_raised = get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE])
        ankle_raised = get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE])
        hip_standing = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        shoulder_same = get_landmark_array(landmarks[PoseLandmark.RIGHT_SHOULDER])

    # --- 1. 脚の挙上角度 ---
    # 腰→足首ベクトルと鉛直方向のなす角
    leg_elevation = compute_angle_with_vertical(hip_raised, ankle_raised)

    # --- 2. 背中のライン角度 ---
    # 軸脚側の腰を基準に、肩がどれだけ前傾しているか
    # 腰→肩ベクトルと鉛直上向き(y負方向)の角度
    vec_torso = shoulder_same - hip_standing
    vertical_up = np.array([0.0, -1.0, 0.0])
    dot_val = np.dot(vec_torso, vertical_up)
    mag = np.linalg.norm(vec_torso)
    if mag > 0:
        cos_val = np.clip(dot_val / mag, -1.0, 1.0)
        back_line_angle = math.degrees(np.arccos(cos_val))
    else:
        back_line_angle = 0.0

    # --- 3. 膝の伸展角度 ---
    # 腰-膝-足首の3点角度 (膝が頂点)
    knee_extension = compute_angle_3points(hip_raised, knee_raised, ankle_raised)

    # --- 4. 肩-腰-脚のアライメント ---
    # 肩-腰(挙脚側)-足首の3点が一直線に近いか
    alignment = compute_angle_3points(shoulder_same, hip_raised, ankle_raised)

    return ArabesqueMetrics(
        leg_elevation_angle=leg_elevation,
        back_line_angle=back_line_angle,
        knee_extension_angle=knee_extension,
        shoulder_hip_leg_alignment=alignment,
        raised_side=raised_side,
    )


def compute_center_of_gravity_stability(
    frames_landmarks: list[list],
) -> dict:
    """
    複数フレームにわたる重心安定性を計算

    重心の近似 = 左右の腰の中点
    各フレームでの重心位置の標準偏差が小さいほど安定

    Args:
        frames_landmarks: フレームごとのランドマークリスト

    Returns:
        {"stability_score": float, "sway_x": float, "sway_y": float}
    """
    centers = []
    for landmarks in frames_landmarks:
        left_hip = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        right_hip = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        center = (left_hip + right_hip) / 2.0
        centers.append(center)

    centers = np.array(centers)
    sway_x = float(np.std(centers[:, 0]))
    sway_y = float(np.std(centers[:, 1]))

    # 安定性スコア: 揺れが小さいほど高得点 (正規化画像座標基準)
    total_sway = math.sqrt(sway_x**2 + sway_y**2)
    stability_score = max(0, 100 - total_sway * 2000)

    return {
        "stability_score": stability_score,
        "sway_x": sway_x,
        "sway_y": sway_y,
    }


def compute_rotation_analysis(
    dense_frames: list[dict],
    source_fps: float,
    video_duration_ms: float,
) -> dict | None:
    """
    ピルエットの回転数と回転速度を算出する

    肩のX座標差 (left_shoulder_x - right_shoulder_x) の時系列から
    ゼロクロスを検出し、回転数を推定する。

    Args:
        dense_frames: process_video_dense() の出力
        source_fps: 元動画のFPS
        video_duration_ms: 動画の総時間(ms)

    Returns:
        回転解析データの辞書。フレームが少なすぎる場合はNone
    """
    if len(dense_frames) < 5:
        return None

    # 1. shoulder_diff の時系列を作成
    timestamps = [f["timestamp_ms"] for f in dense_frames]
    shoulder_diffs = [
        f["left_shoulder_x"] - f["right_shoulder_x"]
        for f in dense_frames
    ]

    # 2. 移動平均でスムージング（3フレーム窓）
    smoothed = []
    for i in range(len(shoulder_diffs)):
        start = max(0, i - 1)
        end = min(len(shoulder_diffs), i + 2)
        smoothed.append(sum(shoulder_diffs[start:end]) / (end - start))

    # 3. ゼロクロス検出
    zero_crossings = []
    for i in range(len(smoothed) - 1):
        if smoothed[i] * smoothed[i + 1] < 0:
            # 線形補間でゼロクロスの正確なタイムスタンプを推定
            ratio = abs(smoothed[i]) / (abs(smoothed[i]) + abs(smoothed[i + 1]))
            crossing_ms = timestamps[i] + ratio * (timestamps[i + 1] - timestamps[i])
            zero_crossings.append(crossing_ms)

    if len(zero_crossings) < 2:
        # 回転が検出されない場合
        return {
            "rotation_count": 0.0,
            "avg_seconds_per_turn": 0.0,
            "rpm": 0.0,
            "rotation_start_ms": 0.0,
            "rotation_end_ms": video_duration_ms,
            "rotation_duration_ms": video_duration_ms,
            "peak_speed_rpm": 0.0,
        }

    # 4. 回転数 = ゼロクロス数 / 2（半回転ごとにゼロクロス）
    rotation_count = len(zero_crossings) / 2.0
    # 0.5刻みに丸める
    rotation_count = round(rotation_count * 2) / 2.0

    # 5. 回転区間の特定
    rotation_start_ms = zero_crossings[0]
    rotation_end_ms = zero_crossings[-1]
    rotation_duration_ms = rotation_end_ms - rotation_start_ms

    # 6. 回転速度の計算
    if rotation_count > 0 and rotation_duration_ms > 0:
        rotation_duration_sec = rotation_duration_ms / 1000.0
        avg_seconds_per_turn = rotation_duration_sec / rotation_count
        rpm = 60.0 / avg_seconds_per_turn if avg_seconds_per_turn > 0 else 0.0
    else:
        avg_seconds_per_turn = 0.0
        rpm = 0.0

    # 7. ピーク速度: 連続するゼロクロスペア間の最短時間から算出
    peak_speed_rpm = 0.0
    if len(zero_crossings) >= 2:
        min_half_turn_ms = float("inf")
        for i in range(len(zero_crossings) - 1):
            gap = zero_crossings[i + 1] - zero_crossings[i]
            if gap > 0:
                min_half_turn_ms = min(min_half_turn_ms, gap)
        if min_half_turn_ms < float("inf") and min_half_turn_ms > 0:
            # 半回転の最短時間 → 1回転の時間 → RPM
            min_full_turn_sec = (min_half_turn_ms * 2) / 1000.0
            peak_speed_rpm = 60.0 / min_full_turn_sec

    return {
        "rotation_count": rotation_count,
        "avg_seconds_per_turn": round(avg_seconds_per_turn, 2),
        "rpm": round(rpm, 1),
        "rotation_start_ms": round(rotation_start_ms, 1),
        "rotation_end_ms": round(rotation_end_ms, 1),
        "rotation_duration_ms": round(rotation_duration_ms, 1),
        "peak_speed_rpm": round(peak_speed_rpm, 1),
    }


# ============================================================
# パ・ド・ドゥ（ペアダンス）専用メトリクス
# ============================================================

@dataclass
class PasDeDeuXMetrics:
    """パ・ド・ドゥ解析結果"""
    shared_com_displacement: float   # 合成重心の変位量（正規化座標）
    com_x: float                     # 合成重心X
    com_y: float                     # 合成重心Y
    within_base: bool                # 合成重心がサポート基底内にあるか
    trunk_verticality: float         # サポートされる側の体幹垂直角度（度）
    support_distance: float          # 腰間距離（正規化座標）
    supported_person_id: int         # サポートされている側のID (0 or 1)

    @property
    def scores(self) -> dict:
        return {
            "shared_com": self._score_shared_com(),
            "trunk_angle": self._score_trunk_angle(),
            "support_distance": self._score_support_distance(),
        }

    def _score_shared_com(self) -> float:
        """合成重心の安定性スコア (変位が小さいほど高得点)"""
        d = self.shared_com_displacement
        if d <= 0.03:
            return 100.0
        if d >= 0.12:
            return 20.0
        return 20.0 + (0.12 - d) / (0.12 - 0.03) * 80.0

    def _score_trunk_angle(self) -> float:
        """体幹垂直性スコア (0度=完璧、15度以上=低得点)"""
        a = self.trunk_verticality
        if a <= 3.0:
            return 100.0
        if a >= 15.0:
            return 25.0
        return 25.0 + (15.0 - a) / (15.0 - 3.0) * 75.0

    def _score_support_distance(self) -> float:
        """サポート距離スコア (理想範囲0.15-0.30で最高得点)"""
        d = self.support_distance
        if 0.15 <= d <= 0.30:
            return 100.0
        if d < 0.15:
            # 近すぎ
            if d <= 0.05:
                return 30.0
            return 30.0 + (d - 0.05) / (0.15 - 0.05) * 70.0
        else:
            # 遠すぎ
            if d >= 0.50:
                return 25.0
            return 25.0 + (0.50 - d) / (0.50 - 0.30) * 75.0


def identify_supported_dancer(landmarks_a: list, landmarks_b: list) -> int:
    """サポートされている側（持ち上げられている側）を判定する。

    足が高い方（y座標が小さい方）がサポートされている側。

    Returns:
        0 (person A) or 1 (person B)
    """
    def avg_foot_height(lms):
        la = lms[PoseLandmark.LEFT_ANKLE]
        ra = lms[PoseLandmark.RIGHT_ANKLE]
        return (la.y + ra.y) / 2.0

    a_foot = avg_foot_height(landmarks_a)
    b_foot = avg_foot_height(landmarks_b)
    # y座標が小さい = 画面上方 = 足が高い = サポートされている
    return 0 if a_foot < b_foot else 1


def compute_shared_center_of_mass(
    landmarks_a: list, landmarks_b: list
) -> dict:
    """2人の合成重心とサポート基底内の判定を計算する。

    Returns:
        {"com_x", "com_y", "base_center_x", "base_center_y",
         "displacement", "within_base"}
    """
    # 各ダンサーのhip中心
    hip_a = (
        get_landmark_array(landmarks_a[PoseLandmark.LEFT_HIP])
        + get_landmark_array(landmarks_a[PoseLandmark.RIGHT_HIP])
    ) / 2.0

    hip_b = (
        get_landmark_array(landmarks_b[PoseLandmark.LEFT_HIP])
        + get_landmark_array(landmarks_b[PoseLandmark.RIGHT_HIP])
    ) / 2.0

    # 合成重心（等重み）
    com = (hip_a + hip_b) / 2.0

    # サポート基底: 4つの足首で形成される範囲の中心
    ankles = [
        get_landmark_array(landmarks_a[PoseLandmark.LEFT_ANKLE]),
        get_landmark_array(landmarks_a[PoseLandmark.RIGHT_ANKLE]),
        get_landmark_array(landmarks_b[PoseLandmark.LEFT_ANKLE]),
        get_landmark_array(landmarks_b[PoseLandmark.RIGHT_ANKLE]),
    ]
    base_center = np.mean(ankles, axis=0)

    # X-Y平面での変位
    displacement = float(math.hypot(
        com[0] - base_center[0], com[1] - base_center[1]
    ))

    # 簡易的な「基底内」判定: 足首のX/Y範囲内にあるか
    xs = [a[0] for a in ankles]
    ys = [a[1] for a in ankles]
    margin = 0.03  # 少し余裕を持たせる
    within_base = (
        min(xs) - margin <= com[0] <= max(xs) + margin
        and min(ys) - margin <= com[1] <= max(ys) + margin
    )

    return {
        "com_x": float(com[0]),
        "com_y": float(com[1]),
        "base_center_x": float(base_center[0]),
        "base_center_y": float(base_center[1]),
        "displacement": round(displacement, 4),
        "within_base": within_base,
    }


def compute_trunk_verticality(landmarks: list) -> float:
    """体幹（肩中心→骨盤中心）の垂直角度を計算する。

    0度 = 完全に垂直, 90度 = 水平

    Returns:
        垂直からの偏差角度（度）
    """
    shoulder_center = (
        get_landmark_array(landmarks[PoseLandmark.LEFT_SHOULDER])
        + get_landmark_array(landmarks[PoseLandmark.RIGHT_SHOULDER])
    ) / 2.0

    hip_center = (
        get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        + get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
    ) / 2.0

    # 骨盤→肩のベクトル（上方向）
    trunk_vec = shoulder_center - hip_center
    # 鉛直上向き: MediaPipeではy軸が下向きなので[0, -1, 0]
    vertical_up = np.array([0.0, -1.0, 0.0])

    dot_val = np.dot(trunk_vec, vertical_up)
    mag = np.linalg.norm(trunk_vec)
    if mag == 0:
        return 0.0

    cos_val = np.clip(dot_val / mag, -1.0, 1.0)
    return float(math.degrees(np.arccos(cos_val)))


def compute_support_distance(landmarks_a: list, landmarks_b: list) -> float:
    """2人のhip中心間のユークリッド距離（正規化座標）。"""
    hip_a = (
        get_landmark_array(landmarks_a[PoseLandmark.LEFT_HIP])
        + get_landmark_array(landmarks_a[PoseLandmark.RIGHT_HIP])
    ) / 2.0

    hip_b = (
        get_landmark_array(landmarks_b[PoseLandmark.LEFT_HIP])
        + get_landmark_array(landmarks_b[PoseLandmark.RIGHT_HIP])
    ) / 2.0

    return float(math.hypot(hip_a[0] - hip_b[0], hip_a[1] - hip_b[1]))


def analyze_pas_de_deux(
    landmarks_a: list, landmarks_b: list
) -> PasDeDeuXMetrics:
    """パ・ド・ドゥの総合解析を行う。

    Args:
        landmarks_a: Person 0 のランドマーク
        landmarks_b: Person 1 のランドマーク

    Returns:
        PasDeDeuXMetrics
    """
    supported_id = identify_supported_dancer(landmarks_a, landmarks_b)
    supported_lms = landmarks_a if supported_id == 0 else landmarks_b

    com_data = compute_shared_center_of_mass(landmarks_a, landmarks_b)
    trunk_angle = compute_trunk_verticality(supported_lms)
    distance = compute_support_distance(landmarks_a, landmarks_b)

    return PasDeDeuXMetrics(
        shared_com_displacement=com_data["displacement"],
        com_x=com_data["com_x"],
        com_y=com_data["com_y"],
        within_base=com_data["within_base"],
        trunk_verticality=trunk_angle,
        support_distance=round(distance, 4),
        supported_person_id=supported_id,
    )
