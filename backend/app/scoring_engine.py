from __future__ import annotations

"""
Ballet Scoring Engine - プロ基準との比較・アドバイス生成

アラベスク・パッセの各ポーズに対して、プロフェッショナル基準と比較し、
0-100のスコアと具体的な改善アドバイスを日本語で返す。

■ パッセ (Passe / Retire) の評価指標:

1. 軸脚の膝伸展 (Standing Knee Extension)
   - 軸脚の腰-膝-足首の角度
   - プロ基準: 175-180度 (完全に真っ直ぐ)

2. パッセ脚の膝角度 (Working Knee Angle)
   - 挙脚側の腰-膝-足首の角度
   - プロ基準: 30-50度 (深く折りたたまれている)

3. パッセ脚のつま先位置 (Toe Placement)
   - つま先が軸脚の膝の高さにあるか
   - 挙脚の足首のy座標と軸脚の膝のy座標の差分

4. 骨盤の水平性 (Pelvic Tilt)
   - 左右の腰の高さの差
   - プロ基準: 差が限りなくゼロに近い (水平)

5. アン・ドゥオール (En Dehors / Turnout)
   - パッセ脚の膝が外側を向いているか
   - 腰-膝ベクトルの水平成分の方向で判定
"""

from dataclasses import dataclass, field
import numpy as np

from .ballet_metrics import (
    PoseLandmark,
    ArabesqueMetrics,
    PasDeDeuXMetrics,
    analyze_arabesque,
    analyze_pas_de_deux,
    compute_angle_3points,
    compute_angle_with_vertical,
    compute_center_of_gravity_stability,
    compute_rotation_analysis,
    get_landmark_array,
)


# ============================================================
# パッセ解析
# ============================================================

@dataclass
class PasseMetrics:
    """パッセ (ルティレ) 解析結果"""
    standing_knee_angle: float     # 軸脚の膝伸展角度
    working_knee_angle: float      # パッセ脚の膝角度
    toe_placement_diff: float      # つま先位置と軸脚膝の高さの差 (正規化座標)
    pelvic_tilt: float             # 骨盤傾斜 (左右腰のy座標差, 正規化)
    turnout_angle: float           # アン・ドゥオール角度
    raised_side: str               # パッセ脚側 ("left" or "right")

    @property
    def scores(self) -> dict:
        return {
            "standing_knee": self._score_standing_knee(),
            "working_knee": self._score_working_knee(),
            "toe_placement": self._score_toe_placement(),
            "pelvic_tilt": self._score_pelvic_tilt(),
            "turnout": self._score_turnout(),
        }

    def _score_standing_knee(self) -> float:
        # 175度以上で満点、150度で30点（マイルド化）
        if self.standing_knee_angle >= 175:
            return 100
        if self.standing_knee_angle <= 150:
            return 30
        return 30 + (self.standing_knee_angle - 150) / 25 * 70

    def _score_working_knee(self) -> float:
        # 25-55度が理想的（幅を広げ）。乖離でスコアを緩やかに下げる
        if 25 <= self.working_knee_angle <= 55:
            return 100
        deviation = min(
            abs(self.working_knee_angle - 25),
            abs(self.working_knee_angle - 55),
        )
        return max(25, 100 - deviation * 2)

    def _score_toe_placement(self) -> float:
        # 差が小さいほど良い（正規化座標で0.03以内なら満点、緩和）
        diff = abs(self.toe_placement_diff)
        if diff <= 0.03:
            return 100
        return max(25, 100 - (diff - 0.03) * 800)

    def _score_pelvic_tilt(self) -> float:
        # 傾きが0に近いほど良い（正規化座標で0.02以内なら満点、緩和）
        tilt = abs(self.pelvic_tilt)
        if tilt <= 0.02:
            return 100
        return max(25, 100 - (tilt - 0.02) * 1000)

    def _score_turnout(self) -> float:
        # 膝が外に開いている角度。90度(真横)で満点、基礎点あり
        if self.turnout_angle >= 80:
            return 100
        return max(20, 20 + self.turnout_angle / 80 * 80)


@dataclass
class PirouetteMetrics:
    """ピルエット解析結果"""
    standing_knee_angle: float       # 軸脚の膝伸展角度
    releve_height: float             # ルルヴェの高さ (heel.y - foot_index.y, 正規化座標)
    vertical_axis_angle: float       # 体幹の垂直性 (肩中心-腰中心の鉛直からの偏差, 度)
    arm_position_score_raw: float    # 腕のポジション品質 (0-1 の生値)
    pelvic_tilt: float               # 骨盤傾斜 (左右腰のy座標差, 正規化)
    working_knee_angle: float        # パッセ脚の膝角度
    raised_side: str                 # パッセ脚側 ("left" or "right")

    @property
    def scores(self) -> dict:
        return {
            "standing_knee": self._score_standing_knee(),
            "releve_height": self._score_releve_height(),
            "vertical_axis": self._score_vertical_axis(),
            "arm_position": self._score_arm_position(),
            "pelvic_level": self._score_pelvic_level(),
            "working_leg": self._score_working_leg(),
        }

    def _score_standing_knee(self) -> float:
        if self.standing_knee_angle >= 175:
            return 100
        if self.standing_knee_angle <= 150:
            return 30
        return 30 + (self.standing_knee_angle - 150) / 25 * 70

    def _score_releve_height(self) -> float:
        # heel.y - foot_index.y: 小さいほど良いルルヴェ
        diff = abs(self.releve_height)
        if diff <= 0.01:
            return 100
        if diff >= 0.06:
            return 20
        return 20 + (0.06 - diff) / 0.05 * 80

    def _score_vertical_axis(self) -> float:
        # 鉛直からの偏差: 0-3度が理想
        if self.vertical_axis_angle <= 3:
            return 100
        if self.vertical_axis_angle >= 15:
            return 25
        return 25 + (15 - self.vertical_axis_angle) / 12 * 75

    def _score_arm_position(self) -> float:
        return max(20, self.arm_position_score_raw * 100)

    def _score_pelvic_level(self) -> float:
        tilt = abs(self.pelvic_tilt)
        if tilt <= 0.02:
            return 100
        return max(25, 100 - (tilt - 0.02) * 1000)

    def _score_working_leg(self) -> float:
        if 25 <= self.working_knee_angle <= 55:
            return 100
        deviation = min(
            abs(self.working_knee_angle - 25),
            abs(self.working_knee_angle - 55),
        )
        return max(25, 100 - deviation * 2)


def detect_passe_leg(landmarks: list) -> str:
    """パッセ脚 = 膝がより高い位置にある側"""
    left_knee_y = landmarks[PoseLandmark.LEFT_KNEE].y
    right_knee_y = landmarks[PoseLandmark.RIGHT_KNEE].y
    return "left" if left_knee_y < right_knee_y else "right"


def analyze_passe(landmarks: list) -> PasseMetrics:
    """パッセ (ルティレ) ポーズを総合解析"""
    raised_side = detect_passe_leg(landmarks)

    if raised_side == "left":
        # パッセ脚 = 左, 軸脚 = 右
        hip_w = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        knee_w = get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE])
        ankle_w = get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE])
        hip_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        knee_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE])
        ankle_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE])
    else:
        hip_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        knee_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE])
        ankle_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE])
        hip_s = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        knee_s = get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE])
        ankle_s = get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE])

    # 1. 軸脚の膝伸展
    standing_knee = compute_angle_3points(hip_s, knee_s, ankle_s)

    # 2. パッセ脚の膝角度
    working_knee = compute_angle_3points(hip_w, knee_w, ankle_w)

    # 3. つま先位置: パッセ脚の足首y と 軸脚の膝y の差 (正規化座標)
    toe_placement_diff = float(ankle_w[1] - knee_s[1])

    # 4. 骨盤の水平性: 左右腰のy座標差
    left_hip = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
    right_hip = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
    pelvic_tilt = float(left_hip[1] - right_hip[1])

    # 5. アン・ドゥオール: パッセ脚の膝の開き
    #    腰→膝ベクトルの水平成分と正面方向のなす角
    vec_thigh = knee_w - hip_w
    # xz平面での開き角度 (z=奥行き方向)
    turnout_angle = float(np.degrees(np.arctan2(abs(vec_thigh[0]), abs(vec_thigh[1]))))

    return PasseMetrics(
        standing_knee_angle=standing_knee,
        working_knee_angle=working_knee,
        toe_placement_diff=toe_placement_diff,
        pelvic_tilt=pelvic_tilt,
        turnout_angle=turnout_angle,
        raised_side=raised_side,
    )


# ============================================================
# ピルエット解析
# ============================================================

def analyze_pirouette(landmarks: list) -> PirouetteMetrics:
    """ピルエットポーズを総合解析"""
    raised_side = detect_passe_leg(landmarks)

    if raised_side == "left":
        hip_w = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        knee_w = get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE])
        ankle_w = get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE])
        hip_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        knee_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE])
        ankle_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE])
        heel_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_HEEL])
        foot_index_s = get_landmark_array(landmarks[PoseLandmark.RIGHT_FOOT_INDEX])
    else:
        hip_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
        knee_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE])
        ankle_w = get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE])
        hip_s = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
        knee_s = get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE])
        ankle_s = get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE])
        heel_s = get_landmark_array(landmarks[PoseLandmark.LEFT_HEEL])
        foot_index_s = get_landmark_array(landmarks[PoseLandmark.LEFT_FOOT_INDEX])

    # 1. 軸脚の膝伸展
    standing_knee = compute_angle_3points(hip_s, knee_s, ankle_s)

    # 2. ルルヴェの高さ: ヒールとつま先のy座標差
    releve_height = float(heel_s[1] - foot_index_s[1])

    # 3. 体幹の垂直性: 肩中心→腰中心ベクトルと鉛直方向の角度
    left_shoulder = get_landmark_array(landmarks[PoseLandmark.LEFT_SHOULDER])
    right_shoulder = get_landmark_array(landmarks[PoseLandmark.RIGHT_SHOULDER])
    left_hip = get_landmark_array(landmarks[PoseLandmark.LEFT_HIP])
    right_hip = get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP])
    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2

    torso_vec = shoulder_center - hip_center
    vertical_up = np.array([0.0, -1.0, 0.0])
    dot_val = np.dot(torso_vec, vertical_up)
    mag = np.linalg.norm(torso_vec)
    if mag > 0:
        cos_val = np.clip(dot_val / mag, -1.0, 1.0)
        vertical_axis_angle = float(np.degrees(np.arccos(cos_val)))
    else:
        vertical_axis_angle = 0.0

    # 4. 腕のポジション: 肘の開きと手首の近さの複合スコア
    left_elbow = get_landmark_array(landmarks[PoseLandmark.LEFT_ELBOW])
    right_elbow = get_landmark_array(landmarks[PoseLandmark.RIGHT_ELBOW])
    left_wrist = get_landmark_array(landmarks[PoseLandmark.LEFT_WRIST])
    right_wrist = get_landmark_array(landmarks[PoseLandmark.RIGHT_WRIST])

    # a) 肘の開き: 肩幅以上に開いているか
    elbow_spread = abs(left_elbow[0] - right_elbow[0])
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    spread_ratio = elbow_spread / max(shoulder_width, 0.01)
    spread_score = min(1.0, spread_ratio)

    # b) 手首の近さ: 腕が丸くまとまっているか
    wrist_dist = float(np.linalg.norm(left_wrist - right_wrist))
    wrist_ratio = wrist_dist / max(shoulder_width, 0.01)
    if wrist_ratio <= 1.0:
        wrist_score = 1.0
    else:
        wrist_score = max(0.0, 1.0 - (wrist_ratio - 1.0) * 0.5)

    arm_position_raw = 0.5 * spread_score + 0.5 * wrist_score

    # 5. 骨盤の水平性
    pelvic_tilt = float(left_hip[1] - right_hip[1])

    # 6. パッセ脚の膝角度
    working_knee = compute_angle_3points(hip_w, knee_w, ankle_w)

    return PirouetteMetrics(
        standing_knee_angle=standing_knee,
        releve_height=releve_height,
        vertical_axis_angle=vertical_axis_angle,
        arm_position_score_raw=arm_position_raw,
        pelvic_tilt=pelvic_tilt,
        working_knee_angle=working_knee,
        raised_side=raised_side,
    )


# ============================================================
# ポーズ自動分類
# ============================================================

def classify_pose(landmarks: list) -> str:
    """
    ランドマークからポーズの種別を自動推定

    判定ロジック:
    - 片脚が後方に高く上がっている → アラベスク
    - 片脚の膝が深く曲がり、つま先が軸脚の膝付近 → パッセ
    - いずれにも該当しない → unknown
    """
    left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]
    left_knee = landmarks[PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[PoseLandmark.RIGHT_KNEE]
    left_hip = landmarks[PoseLandmark.LEFT_HIP]
    right_hip = landmarks[PoseLandmark.RIGHT_HIP]

    # 足首の高さの差 (yが小さいほど高い位置)
    ankle_diff = abs(left_ankle.y - right_ankle.y)

    # 膝の高さの差
    knee_diff = abs(left_knee.y - right_knee.y)

    # 片脚が腰より高い位置にある → アラベスク候補
    higher_ankle_y = min(left_ankle.y, right_ankle.y)
    hip_avg_y = (left_hip.y + right_hip.y) / 2

    if higher_ankle_y < hip_avg_y and ankle_diff > 0.08:
        return "arabesque"

    # 膝の高さに差があり、片方の膝が曲がっている → パッセ/ピルエット候補
    if knee_diff > 0.05:
        raised = "left" if left_knee.y < right_knee.y else "right"
        if raised == "left":
            knee_angle = compute_angle_3points(
                get_landmark_array(landmarks[PoseLandmark.LEFT_HIP]),
                get_landmark_array(landmarks[PoseLandmark.LEFT_KNEE]),
                get_landmark_array(landmarks[PoseLandmark.LEFT_ANKLE]),
            )
            # 軸脚 = 右
            heel_s = landmarks[PoseLandmark.RIGHT_HEEL]
            foot_index_s = landmarks[PoseLandmark.RIGHT_FOOT_INDEX]
        else:
            knee_angle = compute_angle_3points(
                get_landmark_array(landmarks[PoseLandmark.RIGHT_HIP]),
                get_landmark_array(landmarks[PoseLandmark.RIGHT_KNEE]),
                get_landmark_array(landmarks[PoseLandmark.RIGHT_ANKLE]),
            )
            heel_s = landmarks[PoseLandmark.LEFT_HEEL]
            foot_index_s = landmarks[PoseLandmark.LEFT_FOOT_INDEX]

        if knee_angle < 120:
            # パッセ的な脚のポジション → ピルエットかパッセかを判別
            # ルルヴェ判定: ヒールがつま先に近い高さ
            releve_diff = abs(heel_s.y - foot_index_s.y)
            is_on_releve = releve_diff < 0.04

            # 垂直性判定: 肩中心が腰中心の真上
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            vertical_deviation = abs(shoulder_center_x - hip_center_x)
            is_vertical = vertical_deviation < 0.04

            if is_on_releve and is_vertical:
                return "pirouette"

            return "passe"

    return "unknown"


# ============================================================
# アドバイス生成
# ============================================================

@dataclass
class Advice:
    """改善アドバイス1項目"""
    metric_name: str       # 指標名
    score: float           # 0-100
    level: str             # "excellent" / "good" / "needs_work"
    message: str           # 日本語のアドバイス文
    priority: int          # 優先度 (1が最優先)


@dataclass
class AnalysisResult:
    """総合解析結果"""
    pose_type: str                          # "arabesque" / "passe" / "unknown"
    overall_score: float                    # 総合スコア (0-100)
    metrics: dict                           # 各指標の生値
    scores: dict                            # 各指標のスコア (0-100)
    advice: list[Advice] = field(default_factory=list)
    stability: dict | None = None           # 重心安定性 (動画のみ)
    landmarks_data: list[dict] = field(default_factory=list)
    best_frame_index: int = 0               # 最高スコアのフレームインデックス
    best_frame_timestamp_ms: float = 0.0    # 最高スコアのフレームのタイムスタンプ
    total_frames_analyzed: int = 1           # 解析したフレーム数
    video_duration_ms: float = 0.0          # 動画の総時間
    rotation_data: dict | None = None       # 回転解析データ（ピルエット動画のみ）
    pair_data: dict | None = None            # ペア解析データ（パ・ド・ドゥのみ）


def _score_to_level(score: float) -> str:
    if score >= 90:
        return "excellent"
    elif score >= 70:
        return "good"
    return "needs_work"


def _compute_overall(scores: dict) -> float:
    """総合スコア: 平均70% + 中央値30%（1項目低くてもバランス維持）"""
    vals = list(scores.values())
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    sorted_vals = sorted(vals)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        median = (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    else:
        median = sorted_vals[mid]
    return 0.7 * mean + 0.3 * median


def generate_arabesque_advice(metrics: ArabesqueMetrics) -> list[Advice]:
    """アラベスクに対する3ブロック構成のアドバイスを生成"""
    scores = metrics.scores
    advice_list = []

    # ── 足の高さ ──
    s = scores["leg_elevation"]
    angle = metrics.leg_elevation_angle
    target = 90.0
    gap = target - angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"足の高さ{angle:.0f}度 ── 目標の{target:.0f}度をしっかりクリアしています！"
            "足の付け根からの引き上げがとても美しいです。\n\n"
            "【さらに磨くなら】\n"
            "「足先で空に虹を描く」ようなイメージで、"
            "さらに遠くへ伸ばす意識を持ってみましょう。"
            "高さだけでなく「伸びやかさ」が加わると、一段と輝きます。\n\n"
            "【おすすめの練習】\n"
            "アラベスクのまま8カウントキープする練習を。"
            "キープ中も「まだ遠くへ」と足を伸ばし続ける意識が、さらなる美しさにつながります。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"足の高さは{angle:.0f}度。目標の{target:.0f}度まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「後ろの壁を足裏でそっと押し返す」ようなイメージを持ってみましょう。"
            "上に上げるのではなく、遠くへ伸ばす意識にすると、"
            "自然と高さが出てきます。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまってバットマン・タンデュからゆっくり足を上げていく練習を。"
            "ももの裏のストレッチも毎日少しずつ続けると、確実に変わります！"
        )
    else:
        msg = (
            "【現状】\n"
            f"足の高さが{angle:.0f}度で、目標の{target:.0f}度までまだ距離があります。"
            "焦らず一歩ずつ高くしていきましょう！\n\n"
            "【意識するイメージ】\n"
            "「足先で空に虹を描く」ように、"
            "力で持ち上げるのではなく、遠くへスーッと伸ばしていく感覚を大切にしましょう。"
            "高さは後からついてきます。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、低い位置でいいのでまっすぐ美しく足を伸ばすことから始めましょう。"
            "毎日のストレッチで少しずつやわらかくなっていきます！"
        )
    advice_list.append(Advice("足の高さ", s, _score_to_level(s), msg, 1))

    # ── 背中のライン ──
    s = scores["back_line"]
    angle = metrics.back_line_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"背中の角度{angle:.0f}度 ── 理想的なバランスです！"
            "背中がすっと伸びて、肩も自然に下がっている美しいフォームです。\n\n"
            "【さらに磨くなら】\n"
            "「頭のてっぺんを糸で吊られている」ようなイメージで、"
            "さらに上体の伸びを意識してみましょう。"
            "引き上げが加わると、より軽やかで優雅な印象になります。\n\n"
            "【おすすめの練習】\n"
            "このフォームを体に記憶させるために、鏡の前でゆっくり5秒キープを繰り返してみましょう。"
        )
    elif s >= 70:
        if angle < 10:
            msg = (
                "【現状】\n"
                f"背中の角度は{angle:.0f}度で、体がやや直立しすぎています。\n\n"
                "【意識するイメージ】\n"
                "「おへそと背中をそっと近づける」ようにしながら、"
                "少しだけ前へ体を預けてみましょう。"
                "肩からつま先まで1本の美しいラインをイメージしてください。\n\n"
                "【おすすめの練習】\n"
                "鏡の前で横向きに立ち、肩→腰→つま先が一直線になるポイントを探す練習をしてみましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"背中の角度は{angle:.0f}度 ── あともう少しで理想のラインに届きます！\n\n"
                "【意識するイメージ】\n"
                "「頭のてっぺんを糸で吊られている」ように上に伸びながら、"
                "「おへそと背中をそっと近づける」意識を加えてみましょう。"
                "この二つの力で、背中のラインが一気に整います。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまり、カンブレ（上体の前後の動き）をゆっくり丁寧に行うと、"
                "背中のコントロール力がつきます。"
            )
    else:
        if angle > 35:
            msg = (
                "【現状】\n"
                f"背中の角度が{angle:.0f}度で、上体が前に倒れすぎています。\n\n"
                "【意識するイメージ】\n"
                "「おへそと背中をそっと近づける」ようにおなかを引き込みましょう。"
                "前に倒れるのではなく、「上に伸びながら少しだけ前へ」という感覚です。\n\n"
                "【おすすめの練習】\n"
                "バーに両手でつかまり、足を低い位置に保ったまま"
                "上体のバランスだけに集中する練習から始めましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"背中の角度が{angle:.0f}度で、体がまっすぐすぎて足とのバランスが取りにくい状態です。\n\n"
                "【意識するイメージ】\n"
                "「頭のてっぺんを糸で上に引っ張られながら、少しだけ前に体を預ける」"
                "イメージを持ってみましょう。上体と足が互いにバランスを取り合う感覚です。\n\n"
                "【おすすめの練習】\n"
                "鏡を横に見ながら、ゆっくりとカンブレの練習を。"
                "少しずつ「前に傾ける感覚」を体で覚えていきましょう。"
            )
    advice_list.append(Advice("背中のライン", s, _score_to_level(s), msg, 2))

    # ── おひざの伸び ──
    s = scores["knee_extension"]
    angle = metrics.knee_extension_angle
    target = 175.0
    gap = target - angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"ひざの角度{angle:.0f}度 ── まっすぐ美しく伸びています！"
            "ひざ裏からつま先まで、1本のラインが完成しています。\n\n"
            "【さらに磨くなら】\n"
            "「つま先からビームを出す」イメージで、"
            "さらに遠くへ伸ばす意識を持ってみましょう。"
            "伸びの質がさらにワンランク上がります。\n\n"
            "【おすすめの練習】\n"
            "デヴェロッペでゆっくり足を伸ばしきる練習を。"
            "「伸ばしきった瞬間」の感覚を体に染み込ませていきましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"ひざの角度は{angle:.0f}度。まっすぐ（{target:.0f}度）まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「つま先から遠くの空気を押し出す」ようなイメージで伸ばしてみましょう。"
            "力で押すのではなく、ひざの裏からスーッと伸びていく感覚です。\n\n"
            "【おすすめの練習】\n"
            "バットマン・タンデュの時から、ひざをしっかり伸ばしきるクセをつけましょう。"
            "基本の動きが美しければ、アラベスクのひざも自然と伸びてきます。"
        )
    else:
        msg = (
            "【現状】\n"
            f"ひざが少し曲がっています。（現在{angle:.0f}度 / 目標{target:.0f}度）\n\n"
            "【意識するイメージ】\n"
            "「つま先からビームを出す」ようなイメージを持ってみましょう！"
            "「力で押す」のではなく、ももの裏側からスーッと伸びていく感覚です。\n\n"
            "【おすすめの練習】\n"
            "まずはバーに両手でつかまり、ゆっくりと足を後ろへ伸ばす練習から。"
            "高く上げることよりも、まずは「低くてもまっすぐな美しい1本の矢」を"
            "作ることを目標にしてみましょう！"
        )
    advice_list.append(Advice("おひざの伸び", s, _score_to_level(s), msg, 3))

    # ── 体のライン ──
    s = scores["alignment"]
    angle = metrics.shoulder_hip_leg_alignment
    target = 180.0
    gap = abs(target - angle)
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"肩から足先までのラインが{angle:.0f}度 ── ほぼ完璧な一直線です！"
            "肩・腰・つま先が美しくつながった、見事なアラベスクです。\n\n"
            "【さらに磨くなら】\n"
            "「肩から足先まで1本の美しい矢になる」イメージで、"
            "さらに両端に向かって伸びる意識を。"
            "伸びやかさが増すと、客席からの見栄えがまったく変わります。\n\n"
            "【おすすめの練習】\n"
            "鏡の前で横向きにアラベスクを取り、"
            "ラインの美しさを自分の目で確認しながらキープする練習を。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"体のラインは{angle:.0f}度。一直線（{target:.0f}度）まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「肩から足先まで1本の矢になる」イメージで、"
            "肩を下げて腰をまっすぐにする意識を加えてみましょう。"
            "上体と足が引っ張り合うことで、美しいラインが生まれます。\n\n"
            "【おすすめの練習】\n"
            "鏡で横向きに立ち、肩→腰→つま先が一直線になっているか"
            "確認しながらポジションを微調整する練習をしてみましょう。"
        )
    else:
        msg = (
            "【現状】\n"
            f"体のラインが{angle:.0f}度で、肩・腰・足のつながりが崩れています。\n\n"
            "【意識するイメージ】\n"
            "「肩から足先まで1本の美しい矢になる」イメージを持ちましょう。"
            "おなかに力を入れて体幹を安定させると、ラインが一気に整います。\n\n"
            "【おすすめの練習】\n"
            "まずは足を低い位置から始めて、体をまっすぐに保つことを最優先に。"
            "「低くても美しいライン」が作れるようになったら、少しずつ高くしていきましょう！"
        )
    advice_list.append(Advice("体のライン", s, _score_to_level(s), msg, 4))

    # すべて高得点の場合
    if all(v >= 90 for v in scores.values()):
        advice_list.append(Advice(
            "総合評価", min(scores.values()), "excellent",
            "【すばらしい！】\n"
            "足の高さ、背中のライン、ひざの伸び、体のライン ── すべてが高水準です！"
            "とても美しいアラベスクが完成しています。\n\n"
            "【さらに磨くなら】\n"
            "ルルヴェ（つま先立ち）でのアラベスクや、"
            "プロムナード（回転しながらのキープ）にも挑戦してみましょう。"
            "基礎がしっかりしているからこそ、次のステージへ進めます！",
            5,
        ))

    advice_list.sort(key=lambda a: a.priority)
    return advice_list


def generate_passe_advice(metrics: PasseMetrics) -> list[Advice]:
    """パッセ（ルティレ）に対する3ブロック構成のアドバイスを生成"""
    scores = metrics.scores
    advice_list = []

    # ── 軸足のひざ ──
    s = scores["standing_knee"]
    angle = metrics.standing_knee_angle
    target = 180.0
    gap = target - angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"軸足のひざ{angle:.0f}度 ── しっかりまっすぐ伸びて、安定した軸ができています！"
            "足の裏で床をしっかり踏めている、とても良いフォームです。\n\n"
            "【さらに磨くなら】\n"
            "「大地にしっかり根を張った大きな木」のイメージを持ちましょう。"
            "根っこ（足裏）が安定しているからこそ、幹（軸足）がまっすぐ伸び、"
            "枝（上げた足や腕）が自由に動けます。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセで8カウントキープ。"
            "「木のように揺るがない」安定感を、さらに磨いていきましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"軸足のひざは{angle:.0f}度。まっすぐ（{target:.0f}度）まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「大地にしっかり根を張った木」のように、足裏で床を踏みしめながら"
            "上に伸びていく意識を持ちましょう。"
            "「後ろに押す」のではなく「上に伸びる」── この違いがとても大切です。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、ゆっくりルルヴェの上げ下げを繰り返しましょう。"
            "「上がるたびにひざが伸びる」感覚をつかんでいけます。"
        )
    else:
        msg = (
            "【現状】\n"
            f"軸足のひざが曲がっています。（現在{angle:.0f}度 / 目標{target:.0f}度）\n\n"
            "【意識するイメージ】\n"
            "「大地にしっかり根を張った1本の木」になるイメージです。"
            "おなかを上に引き上げながら、足裏で床を押し、"
            "体全体で上に伸びていく感覚を大切にしましょう。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、プリエからルルヴェまでの動きをゆっくり丁寧に。"
            "「立ち上がる瞬間にひざを伸ばしきる」ことを意識してみてください。"
        )
    advice_list.append(Advice("軸足のおひざ", s, _score_to_level(s), msg, 1))

    # ── 上げている足のひざ ──
    s = scores["working_knee"]
    angle = metrics.working_knee_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"上げた足のひざが{angle:.0f}度 ── コンパクトに折りたためていて、"
            "とても綺麗なパッセの形です！\n\n"
            "【さらに磨くなら】\n"
            "「足の三角形をコンパクトなダイヤモンド」にするイメージで、"
            "つま先と軸足ひざの接点をより意識してみましょう。"
            "小さく、でもキラリと光る形を目指して。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセでキープしながら、上げた足の形を鏡で確認する練習を。"
            "「美しいダイヤモンド」の形を体に記憶させていきましょう。"
        )
    elif s >= 70:
        if angle > 55:
            msg = (
                "【現状】\n"
                f"上げた足のひざの角度は{angle:.0f}度で、やや開きすぎています。\n\n"
                "【意識するイメージ】\n"
                "「足でコンパクトなダイヤモンドを作る」イメージです。"
                "ももを上に持ち上げてから、つま先を軸足のひざの横にそっと添えましょう。"
                "大きく開くよりも、小さくまとめた方が美しく見えます。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまり、ゆっくりとパッセのポジションを取る練習を。"
                "「上げる→たたむ」の二段階を意識してみましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"上げた足のひざの角度は{angle:.0f}度 ── あと少しで理想の形です！\n\n"
                "【意識するイメージ】\n"
                "「足でコンパクトなダイヤモンドを作る」イメージで、"
                "ももを上に持ち上げて、つま先を軸足のひざの横に"
                "そっと添えてみましょう。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまり、ゆっくりパッセの形を作る練習を繰り返しましょう。"
                "正しいポジションの「感覚」を体で覚えていけます。"
            )
    else:
        if angle > 55:
            msg = (
                "【現状】\n"
                f"上げた足のひざが{angle:.0f}度で、かなり開いてしまっています。\n\n"
                "【意識するイメージ】\n"
                "「足でコンパクトなダイヤモンドを作る」イメージを持ちましょう。"
                "大きく開くのではなく、小さく美しくまとめることが大切です。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまって、つま先で軸足の内側を"
                "くるぶしから滑らせながら上に持ち上げる練習を。"
                "正しい軌道を体に覚えさせましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"上げた足のひざが{angle:.0f}度で、曲がりすぎています。\n\n"
                "【意識するイメージ】\n"
                "「足の付け根から扇を開く」ように、ひざを横へ導いてみましょう。"
                "ターンアウトの意識で開くと、自然と綺麗な形になります。\n\n"
                "【おすすめの練習】\n"
                "横向きに寝てひざを開閉する運動で、足の付け根をやわらかくしていきましょう。"
                "付け根がやわらかくなると、パッセの形がぐんと良くなります。"
            )
    advice_list.append(Advice("上げた足のおひざ", s, _score_to_level(s), msg, 2))

    # ── つま先の位置 ──
    s = scores["toe_placement"]
    diff = metrics.toe_placement_diff
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "つま先が軸足のひざの横にぴったり ── 理想的なポジションです！"
            "高さもちょうど良く、とても美しいパッセです。\n\n"
            "【さらに磨くなら】\n"
            "「ひざの横にそっと花を添える」ような繊細さを意識してみましょう。"
            "力で押し付けるのではなく、軽やかにそっと。\n\n"
            "【おすすめの練習】\n"
            "目を閉じてパッセを取り、正しい位置に来ているか感覚だけで確認する練習を。"
            "体の感覚だけで正しい位置が分かるようになれば、本物です！"
        )
    elif s >= 70:
        diff_direction = "少し低め" if diff > 0 else "少し高め"
        if diff > 0:
            msg = (
                "【現状】\n"
                f"つま先の位置が{diff_direction}です。もう少し引き上げてみましょう。\n\n"
                "【意識するイメージ】\n"
                "「ひざの横にそっと花を添える」イメージで、"
                "つま先をちょうどひざの横まで持ち上げてみましょう。"
                "足全体を上げるのではなく、つま先を「そっと置く」感覚です。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまって、ゆっくり足を引き上げてひざの横で止める練習を。"
                "「ちょうどいい高さ」の感覚をつかんでいきましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"つま先の位置が{diff_direction}で、ひざより上に行っています。\n\n"
                "【意識するイメージ】\n"
                "「ひざの横にそっと花を添える」── ちょうど横、がポイントです。"
                "上げすぎると体が傾きやすくなるので、ひざの横を狙いましょう。\n\n"
                "【おすすめの練習】\n"
                "鏡を見ながら、つま先がちょうどひざの横に来る位置を確認してみてください。"
                "正しい位置を目で確認すると、体も覚えていきます。"
            )
    else:
        diff_direction = "低い" if diff > 0 else "高い"
        msg = (
            "【現状】\n"
            f"つま先が軸足のひざより{diff_direction}位置にあり、ずれが大きい状態です。\n\n"
            "【意識するイメージ】\n"
            "「ひざの横にそっと花を添える」イメージで、"
            "ちょうどいい高さを見つけていきましょう。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまり、つま先で軸足の内側を"
            "くるぶし→ふくらはぎ→ひざの順にゆっくり滑らせる練習を。"
            "この道順を体で覚えると、正しい位置に自然と止まれるようになります。"
        )
    advice_list.append(Advice("つま先の位置", s, _score_to_level(s), msg, 3))

    # ── 腰の水平 ──
    s = scores["pelvic_tilt"]
    tilt_abs = abs(metrics.pelvic_tilt)
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "腰がまっすぐ水平 ── 足を上げても全くぶれない、素晴らしい安定感です！"
            "体幹の力がしっかり使えています。\n\n"
            "【さらに磨くなら】\n"
            "「腰の上にお盆を載せて、水をこぼさない」イメージで、"
            "さらに繊細なコントロールを意識してみましょう。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセでのキープ時間を延ばしていきましょう。"
            "長くキープしても腰が傾かない力を養えます。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "腰がほんの少しだけ傾いています。あと少しで水平になります！\n\n"
            "【意識するイメージ】\n"
            "「腰の上にお盆を載せて、水をこぼさない」イメージを持ちましょう。"
            "足を上げる時こそ、腰が傾かないようにおなかを引き上げることが大切です。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまってパッセを取り、鏡で左右の腰の高さが揃っているか確認しましょう。"
            "「おなかの引き上げ」がポイントです。"
        )
    else:
        msg = (
            "【現状】\n"
            "腰が傾いてしまっています。足を上げる時に体が横に倒れていませんか？\n\n"
            "【意識するイメージ】\n"
            "「腰の上にお盆を載せて、水をこぼさない」── "
            "足を上げることよりも、まずはこのイメージを最優先にしましょう。"
            "おなかに力を入れて、体を真上に引き上げる意識です。\n\n"
            "【おすすめの練習】\n"
            "足を低い位置で上げるところから始めて、腰がまっすぐのまま"
            "キープできる高さを探しましょう。そこから少しずつ高くしていけば大丈夫です。"
        )
    advice_list.append(Advice("腰の水平", s, _score_to_level(s), msg, 1))

    # ── ターンアウト ──
    s = scores["turnout"]
    angle = metrics.turnout_angle
    target = 80.0
    gap = target - angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"ターンアウト{angle:.0f}度 ── 足の付け根から美しく開いています！"
            "ひざとつま先の向きもしっかり揃っています。\n\n"
            "【さらに磨くなら】\n"
            "「足の付け根から扇をさらに広げる」イメージで、"
            "ターンアウトの質をさらに高めていきましょう。"
            "開き方が自然であるほど、美しく見えます。\n\n"
            "【おすすめの練習】\n"
            "ロン・ド・ジャンブでターンアウトをキープしながら"
            "足を回す練習を。動きの中でも開きを保つ力がつきます。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"ターンアウトは{angle:.0f}度。目標の{target:.0f}度まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「足の付け根から扇を開く」ようにイメージしましょう。"
            "ひざだけで無理に開くのではなく、付け根から自然に回していく感覚です。"
            "おしりの奥の方にそっと力を入れるのがコツです。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、1番ポジションでプリエをする時に"
            "ターンアウトをしっかり意識しましょう。基本のポジションが開きの土台になります。"
        )
    else:
        msg = (
            "【現状】\n"
            f"ターンアウトが{angle:.0f}度で、もう少し開きが欲しいところです。（目標{target:.0f}度）\n\n"
            "【意識するイメージ】\n"
            "「足の付け根から扇を開く」イメージを大切にしましょう。"
            "ひざだけで開こうとすると痛めてしまうので、必ず付け根から回す意識で。\n\n"
            "【おすすめの練習】\n"
            "横向きに寝て、ひざをゆっくり開閉する運動がおすすめです。"
            "足の付け根がやわらかくなると、自然とターンアウトが広がっていきます。"
        )
    advice_list.append(Advice("足の開き", s, _score_to_level(s), msg, 2))

    if all(v >= 90 for v in scores.values()):
        advice_list.append(Advice(
            "総合評価", min(scores.values()), "excellent",
            "【すばらしい！】\n"
            "軸足の安定、パッセの形、腰の水平、ターンアウト ── すべて高水準です！"
            "とても美しいパッセが完成しています。\n\n"
            "【さらに磨くなら】\n"
            "ルルヴェでのパッセや、ピルエットへのステップアップに挑戦してみましょう。"
            "この美しい基礎があれば、きっと回転も上手くいきます！",
            5,
        ))

    advice_list.sort(key=lambda a: a.priority)
    return advice_list


def generate_pirouette_advice(metrics: PirouetteMetrics) -> list[Advice]:
    """ピルエットに対する3ブロック構成のアドバイスを生成"""
    scores = metrics.scores
    advice_list = []

    # ── 軸足のひざ ──
    s = scores["standing_knee"]
    angle = metrics.standing_knee_angle
    target = 180.0
    gap = target - angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"軸足のひざ{angle:.0f}度 ── ルルヴェのまましっかり伸びて、"
            "回転のための美しい軸ができています！\n\n"
            "【さらに磨くなら】\n"
            "「コマの軸のようにまっすぐ1本」のイメージで、"
            "回転中もひざが緩まない強さを意識しましょう。"
            "軸が安定するほど、回転が美しくなります。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセで16カウントキープ。"
            "「コマの軸」のようにぶれない強さを鍛えていきましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"軸足のひざは{angle:.0f}度。まっすぐ（{target:.0f}度）まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「コマの軸のようにまっすぐ1本」── "
            "プリエから立ち上がる瞬間に、一気にひざを伸ばしきるのがポイントです。"
            "回転中もこの「1本の軸」を崩さない意識を持ちましょう。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、プリエ→ルルヴェ・パッセを繰り返しましょう。"
            "「立ち上がった瞬間にひざが伸びきる」感覚をつかむことが大切です。"
        )
    else:
        msg = (
            "【現状】\n"
            f"軸足のひざが曲がっています。（現在{angle:.0f}度 / 目標{target:.0f}度）"
            "ひざが曲がったまま回ると、軸がぶれてしまいます。\n\n"
            "【意識するイメージ】\n"
            "「コマの軸」を思い浮かべてください。"
            "曲がったコマはうまく回りませんよね。"
            "おなかを上に引き上げながら、まっすぐ1本の軸を作る意識を。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、プリエからルルヴェの上げ下げを。"
            "まずは回転なしで「まっすぐ立つ」ことを完璧にしましょう。"
        )
    advice_list.append(Advice("軸足のおひざ", s, _score_to_level(s), msg, 1))

    # ── ルルヴェの高さ ──
    s = scores["releve_height"]
    releve = abs(metrics.releve_height)
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "ルルヴェの高さが十分に出ています！"
            "かかとがしっかり上がり、足の甲も美しく伸びています。\n\n"
            "【さらに磨くなら】\n"
            "「天井から見えない糸で引き上げられている」イメージで、"
            "ふくらはぎだけでなく、体全体で上に伸びる感覚を大切に。"
            "引き上げが強いほど、回転が軽やかになります。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェでのパッセ・キープ時間をどんどん延ばしていきましょう。"
            "高さを保ったまま長くキープできる力が、ピルエットの質を上げます。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "ルルヴェの高さはまずまずですが、もう少し欲しいところです。\n\n"
            "【意識するイメージ】\n"
            "「天井から見えない糸で引き上げられている」ようなイメージで、"
            "ふくらはぎだけで踏ん張るのではなく、"
            "体全体を上に引き上げる意識を持ちましょう。"
            "足の甲を前に押し出すと、さらに高さが出ます。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、8カウントでルルヴェをキープする練習を。"
            "「上に引き上げ続ける」意識が、高さにつながります。"
        )
    else:
        msg = (
            "【現状】\n"
            "ルルヴェの高さがまだ足りない状態です。"
            "かかとがしっかり上がりきっていません。\n\n"
            "【意識するイメージ】\n"
            "「天井から糸で引き上げられる」ように、"
            "体全体で上に伸びていく感覚を持ちましょう。"
            "足だけで頑張ろうとせず、おなかから引き上げることが大切です。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、ルルヴェの上げ下げをたくさん繰り返しましょう。"
            "足の裏をテニスボールでほぐすのも効果的です。"
        )
    advice_list.append(Advice("つま先立ちの高さ", s, _score_to_level(s), msg, 2))

    # ── 体の垂直性 ──
    s = scores["vertical_axis"]
    angle = metrics.vertical_axis_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"体の傾き{angle:.0f}度 ── ほぼ完璧にまっすぐです！"
            "肩→腰→足が一直線になった、回転してもぶれない美しい軸です。\n\n"
            "【さらに磨くなら】\n"
            "「体全体が1本の美しい柱になる」イメージで、"
            "頭のてっぺんから天井に向かって伸び続ける意識を。"
            "この「伸び」が回転の質をさらに高めます。\n\n"
            "【おすすめの練習】\n"
            "スポット（視線の固定点）を決めて実際にピルエットを。"
            "この美しい軸を回転の中でも保つ練習をしていきましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"体の傾きは{angle:.0f}度 ── あと少しでまっすぐになります！\n\n"
            "【意識するイメージ】\n"
            "「体全体が1本の美しい柱になる」イメージです。"
            "おなかを引き込んで、頭のてっぺんが天井に向かって伸びていく感覚を"
            "持ちましょう。回る前に「まっすぐ立てているか」を鏡で確認してみてください。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、ルルヴェ・パッセのまままっすぐ立つ練習を。"
            "「柱のようにまっすぐ」の感覚を体に覚えさせましょう。"
        )
    else:
        msg = (
            "【現状】\n"
            f"体が{angle:.0f}度傾いています。"
            "この状態で回ると、コントロールが効きません。\n\n"
            "【意識するイメージ】\n"
            "「体全体が1本の柱」になるイメージです。"
            "傾いた柱は倒れてしまいますよね。"
            "まずは回転の前に、まっすぐ立つことを完璧にしましょう。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、ルルヴェ・パッセでまっすぐ立つ練習から。"
            "鏡で肩→腰→足が一直線になっているか、しっかり確認してみてください。"
        )
    advice_list.append(Advice("体のまっすぐさ", s, _score_to_level(s), msg, 1))

    # ── アームスのポジション ──
    s = scores["arm_position"]
    arm_raw = metrics.arm_position_score_raw
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "アームスのポジションがとても美しいです！"
            "肩が自然に下がり、ひじが程よい高さで、丸みのある綺麗な形です。\n\n"
            "【さらに磨くなら】\n"
            "「大きなボールをそっと抱える」ようなイメージで、"
            "指先までやわらかさを意識してみましょう。"
            "アームスが美しいと、回転全体の印象が格段に上がります。\n\n"
            "【おすすめの練習】\n"
            "ポール・ド・ブラ（腕の動き）をゆっくり丁寧に練習しましょう。"
            "アン・ナヴァンでのキープ力が、ピルエットの安定につながります。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "アームスのポジションはまずまずですが、もう少し磨けます！\n\n"
            "【意識するイメージ】\n"
            "「大きなボールをそっと抱える」イメージで、"
            "肩を下げて、ひじが落ちないように前に丸くキープしてみましょう。"
            "手首の力を抜いて、指先をやわらかく前へ。\n\n"
            "【おすすめの練習】\n"
            "鏡の前でアン・ナヴァンの形を取り、"
            "「ボールを抱えたまま5秒キープ」を繰り返しましょう。"
            "正しい形を体に記憶させていけます。"
        )
    else:
        msg = (
            "【現状】\n"
            "アームスのポジションが崩れてしまっています。"
            "ひじが下がったり、肩に力が入っていませんか？\n\n"
            "【意識するイメージ】\n"
            "「大きなボールをそっと抱える」── まずはこのイメージから始めましょう。"
            "肩の力を抜いて、ひじを前に、手のひらを自分に向けるように。"
            "力を入れるのではなく「形を保つ」意識です。\n\n"
            "【おすすめの練習】\n"
            "アン・バーからアン・ナヴァン、アン・オーまでの"
            "ポール・ド・ブラをゆっくり丁寧に繰り返しましょう。"
            "腕の通り道を体で覚えることが第一歩です。"
        )
    advice_list.append(Advice("うでのかたち", s, _score_to_level(s), msg, 3))

    # ── 腰の水平 ──
    s = scores["pelvic_level"]
    tilt_abs = abs(metrics.pelvic_tilt)
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "腰がまっすぐ水平 ── 回転しても軸がぶれない、素晴らしい安定感です！\n\n"
            "【さらに磨くなら】\n"
            "「腰にフラフープを水平に回す」イメージで、"
            "回転中もこの水平を保ち続ける意識を持ちましょう。"
            "安定した腰が、美しい回転の土台になります。\n\n"
            "【おすすめの練習】\n"
            "実際にピルエットを回りながら、腰の水平を保つ練習を。"
            "回っている間も「フラフープが水平」のイメージを忘れずに。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "腰がほんの少し傾いています。回転のぶれにつながるので注意！\n\n"
            "【意識するイメージ】\n"
            "「腰にフラフープを水平に回す」イメージを持ちましょう。"
            "足を上げた時こそ、おなかに力を入れて"
            "腰を水平にキープすることが大切です。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまり、パッセを取った状態で"
            "鏡で左右の腰の高さが揃っているか確認しましょう。"
            "おなかの引き上げを意識すると、腰が安定してきます。"
        )
    else:
        msg = (
            "【現状】\n"
            "腰が傾いてしまい、回転が不安定になっています。\n\n"
            "【意識するイメージ】\n"
            "「腰にフラフープを水平に回す」── "
            "フラフープが傾いたら落ちてしまいますよね。"
            "足を上げることよりも、まずは腰の水平を最優先にしましょう。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、腰をまっすぐに保ったまま"
            "パッセの練習を。おなかの力で腰を支える感覚をつかみましょう。"
        )
    advice_list.append(Advice("腰の水平", s, _score_to_level(s), msg, 2))

    # ── 上げた足の形 ──
    s = scores["working_leg"]
    angle = metrics.working_knee_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"上げた足のひざが{angle:.0f}度 ── コンパクトに折りたためていて、"
            "回転しやすい美しいルティレの形です！\n\n"
            "【さらに磨くなら】\n"
            "「三角形をキュッとコンパクトに折りたたむ」ように、"
            "さらに小さくまとめる意識を持ちましょう。"
            "コンパクトなほど、回転が速く美しくなります。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセでキープしながら、"
            "上げた足の形を鏡でチェック。"
            "回転中もこの形を崩さない力をつけていきましょう。"
        )
    elif s >= 70:
        if angle > 55:
            msg = (
                "【現状】\n"
                f"上げた足のひざは{angle:.0f}度で、やや開きすぎています。\n\n"
                "【意識するイメージ】\n"
                "「三角形をキュッとコンパクトに折りたたむ」イメージです。"
                "大きく開いた足よりも、小さくまとまった足の方が"
                "回転が速く、美しく見えます。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまり、つま先を軸足のひざの横にしっかりつける練習を。"
                "「コンパクト」を意識するだけで、回転の質が変わります。"
            )
        else:
            msg = (
                "【現状】\n"
                f"上げた足のひざは{angle:.0f}度 ── あと少しで理想の形です！\n\n"
                "【意識するイメージ】\n"
                "「三角形をキュッとコンパクトに」── "
                "ももを上に持ち上げて、つま先を軸足のひざの横に"
                "しっかりつけるイメージです。\n\n"
                "【おすすめの練習】\n"
                "バーでルルヴェ・パッセの形を取り、正しいポジションでキープする練習を。"
                "形が安定すれば、回転も安定してきます。"
            )
    else:
        if angle > 55:
            msg = (
                "【現状】\n"
                f"上げた足のひざが{angle:.0f}度で、かなり開いてしまっています。"
                "この状態では回転がぶれやすくなります。\n\n"
                "【意識するイメージ】\n"
                "「三角形をキュッとコンパクトに折りたたむ」ことを最優先に。"
                "大きく開くのではなく、小さくまとめることで回転が安定します。\n\n"
                "【おすすめの練習】\n"
                "バーにつかまって、つま先で軸足の内側を滑らせながら"
                "上に持ち上げる練習を。正しい軌道を体に覚えさせましょう。"
            )
        else:
            msg = (
                "【現状】\n"
                f"上げた足のひざが{angle:.0f}度で、曲がりすぎています。\n\n"
                "【意識するイメージ】\n"
                "「足の付け根から扇を開く」ように、"
                "ターンアウトの意識でひざを横へ導いてみましょう。"
                "ひざが横を向くと、自然と綺麗なルティレの形になります。\n\n"
                "【おすすめの練習】\n"
                "横向きに寝てひざを開閉する運動で、付け根をやわらかくしましょう。"
                "付け根が開くようになると、パッセの形が格段に良くなります。"
            )
    advice_list.append(Advice("上げた足のかたち", s, _score_to_level(s), msg, 3))

    # すべて高得点の場合
    if all(v >= 90 for v in scores.values()):
        advice_list.append(Advice(
            "総合評価", min(scores.values()), "excellent",
            "【すばらしい！】\n"
            "ルルヴェの高さ、体の垂直性、アームス、腰の安定 ── すべて高水準です！"
            "とても美しいピルエットのフォームが完成しています。\n\n"
            "【さらに磨くなら】\n"
            "ダブル・ピルエットや、アン・ドゥオール（外回り）にも挑戦してみましょう。"
            "この美しい基礎があれば、きっと回転数も伸びていきます！",
            5,
        ))

    advice_list.sort(key=lambda a: a.priority)
    return advice_list


# ============================================================
# 統合解析エントリポイント
# ============================================================

def evaluate_frame(landmarks: list) -> AnalysisResult:
    """
    1フレームのランドマークから総合評価を行う

    ポーズを自動分類し、該当するメトリクス計算・スコアリング・アドバイス生成を実行
    """
    pose_type = classify_pose(landmarks)

    if pose_type == "arabesque":
        metrics_obj = analyze_arabesque(landmarks)
        scores = metrics_obj.scores
        overall = _compute_overall(scores)
        advice = generate_arabesque_advice(metrics_obj)
        metrics_dict = {
            "leg_elevation_angle": metrics_obj.leg_elevation_angle,
            "back_line_angle": metrics_obj.back_line_angle,
            "knee_extension_angle": metrics_obj.knee_extension_angle,
            "shoulder_hip_leg_alignment": metrics_obj.shoulder_hip_leg_alignment,
            "raised_side": metrics_obj.raised_side,
        }

    elif pose_type == "passe":
        metrics_obj = analyze_passe(landmarks)
        scores = metrics_obj.scores
        overall = _compute_overall(scores)
        advice = generate_passe_advice(metrics_obj)
        metrics_dict = {
            "standing_knee_angle": metrics_obj.standing_knee_angle,
            "working_knee_angle": metrics_obj.working_knee_angle,
            "toe_placement_diff": metrics_obj.toe_placement_diff,
            "pelvic_tilt": metrics_obj.pelvic_tilt,
            "turnout_angle": metrics_obj.turnout_angle,
            "raised_side": metrics_obj.raised_side,
        }

    elif pose_type == "pirouette":
        metrics_obj = analyze_pirouette(landmarks)
        scores = metrics_obj.scores
        overall = _compute_overall(scores)
        advice = generate_pirouette_advice(metrics_obj)
        metrics_dict = {
            "standing_knee_angle": metrics_obj.standing_knee_angle,
            "releve_height": metrics_obj.releve_height,
            "vertical_axis_angle": metrics_obj.vertical_axis_angle,
            "arm_position_score_raw": metrics_obj.arm_position_score_raw,
            "pelvic_tilt": metrics_obj.pelvic_tilt,
            "working_knee_angle": metrics_obj.working_knee_angle,
            "raised_side": metrics_obj.raised_side,
        }

    else:
        scores = {}
        overall = 0.0
        advice = [Advice(
            "ポーズ検出",
            0,
            "needs_work",
            "アラベスク、パッセ、またはピルエットのポーズを検出できませんでした。"
            "カメラの前で横向きに立ち、全身が映るように撮影してください。",
            1,
        )]
        metrics_dict = {}

    return AnalysisResult(
        pose_type=pose_type,
        overall_score=round(overall, 1),
        metrics=metrics_dict,
        scores=scores,
        advice=advice,
    )


def evaluate_video(
    frames_landmarks: list[list],
    frame_indices: list[int] | None = None,
    frame_timestamps: list[float] | None = None,
    video_duration_ms: float = 0.0,
    dense_frames: list[dict] | None = None,
    source_fps: float = 30.0,
) -> AnalysisResult:
    """
    複数フレームの動画解析

    最も代表的なフレーム(最高スコア)を基準に評価し、
    重心安定性を追加で計算する。
    ベストフレームのインデックスとタイムスタンプも記録する。
    """
    if not frames_landmarks:
        return AnalysisResult(
            pose_type="unknown",
            overall_score=0.0,
            metrics={},
            scores={},
            advice=[Advice("入力エラー", 0, "needs_work", "ポーズを検出できるフレームがありませんでした。", 1)],
        )

    _indices = frame_indices or list(range(len(frames_landmarks)))
    _timestamps = frame_timestamps or [0.0] * len(frames_landmarks)

    # 各フレームを評価し、最高スコアのフレームを代表とする
    best_result = None
    best_score = -1.0
    best_idx = 0

    for i, landmarks in enumerate(frames_landmarks):
        result = evaluate_frame(landmarks)
        if result.overall_score > best_score:
            best_score = result.overall_score
            best_result = result
            best_idx = i

    # ベストフレーム情報を記録
    best_result.best_frame_index = _indices[best_idx]
    best_result.best_frame_timestamp_ms = _timestamps[best_idx]
    best_result.total_frames_analyzed = len(frames_landmarks)
    best_result.video_duration_ms = video_duration_ms

    # 重心安定性を追加
    stability = compute_center_of_gravity_stability(frames_landmarks)
    best_result.stability = stability

    # 安定性のアドバイス（ポジティブな表現）
    if stability["stability_score"] < 70:
        best_result.advice.append(Advice(
            "重心の安定性",
            stability["stability_score"],
            _score_to_level(stability["stability_score"]),
            f"重心の安定性スコアは {stability['stability_score']:.0f}/100 です。"
            "ポーズの保持中にわずかな揺れが見られますが、これは自然なことです。"
            "ドゥミ・プリエからゆっくりポーズに入り、"
            "体幹の引き上げを意識すると、さらに安定感が増します。",
            0,
        ))
        best_result.advice.sort(key=lambda a: a.priority)

    # ピルエットの回転解析（dense_framesがある場合のみ）
    if dense_frames and best_result.pose_type == "pirouette":
        rotation_data = compute_rotation_analysis(
            dense_frames, source_fps, video_duration_ms
        )
        best_result.rotation_data = rotation_data

    return best_result


# ============================================================
# パ・ド・ドゥ（ペアダンス）解析
# ============================================================

def generate_pdd_advice(metrics: PasDeDeuXMetrics) -> list[Advice]:
    """パ・ド・ドゥ専用アドバイスを生成する（3ブロック形式・温かい表現）"""
    advice_list = []
    scores = metrics.scores

    # --- 1. 共有重心の安定性 ---
    s = scores["shared_com"]
    level = _score_to_level(s)
    disp = metrics.shared_com_displacement

    if level == "excellent":
        msg = (
            "【すばらしい！】\n"
            f"二人の合成重心がサポート基底の中心付近に安定しています。（変位{disp:.3f}）\n"
            "二人で「一本の大きな木」のようにしっかり根を張っている、見事なバランスです！\n\n"
            "【さらに磨くなら】\n"
            "「二人の体の中心にひとつの光の柱が通っている」イメージで、\n"
            "お互いの呼吸を合わせながら、さらに静かなバランスを追求してみましょう。\n\n"
            "【おすすめの練習】\n"
            "目を閉じて、パートナーの手のひらの温度だけを感じながら\n"
            "プロムナードをゆっくり行う練習がおすすめです。"
        )
    elif level == "good":
        msg = (
            "【現状】\n"
            f"合成重心の変位は{disp:.3f}。おおむね安定していますが、\n"
            "もう少し二人の中心を揃えると、さらに美しいバランスが生まれます。\n\n"
            "【意識するイメージ】\n"
            "「二人の間にシャボン玉がふわりと浮かんでいる」イメージを持ってみましょう。\n"
            "そのシャボン玉を壊さないように、やさしく重心を預け合う感覚です。\n\n"
            "【おすすめの練習】\n"
            "バーを使わず、お互いの手だけでバランスを取るパッセの練習から始めましょう。\n"
            "相手を引っ張らず、そっと支え合う距離感をつかむことが大切です。"
        )
    else:
        msg = (
            "【現状】\n"
            f"合成重心がサポート基底の中心からやや外れています。（変位{disp:.3f}）\n"
            "どちらかが寄りかかりすぎている、または距離が合っていない可能性があります。\n\n"
            "【意識するイメージ】\n"
            "「二人で一つのシーソーに乗っている」と想像してください。\n"
            "どちらか一方に体重が偏ると傾いてしまいます。\n"
            "お互いが自分の軸を保ちながら、そっと手で繋がっている感覚を大切にしましょう。\n\n"
            "【おすすめの練習】\n"
            "まずは二人で向かい合い、両手を合わせて同時にプリエ→ルルヴェ。\n"
            "「引っ張らない・押さない」を意識して、息を合わせることから始めましょう。"
        )

    advice_list.append(Advice(
        metric_name="共有重心の安定性",
        score=round(s, 1),
        level=level,
        message=msg,
        priority=1,
    ))

    # --- 2. 体幹の垂直性 ---
    s = scores["trunk_angle"]
    level = _score_to_level(s)
    angle = metrics.trunk_verticality

    if level == "excellent":
        msg = (
            "【すばらしい！】\n"
            f"サポートを受けながらも体幹が美しく垂直に保たれています。（{angle:.1f}度）\n"
            "パートナーへの信頼と自分自身の軸の強さが見事に調和しています！\n\n"
            "【さらに磨くなら】\n"
            "「天井から見えない糸で頭を引き上げられている」感覚を持ちながら、\n"
            "サポートの手にはほんの少しだけ触れている ── そんな軽やかさを目指しましょう。\n\n"
            "【おすすめの練習】\n"
            "パートナーの手に「小鳥がとまるくらいの軽さ」で触れながら\n"
            "アティチュード・プロムナードを行う練習です。"
        )
    elif level == "good":
        msg = (
            "【現状】\n"
            f"体幹角度は{angle:.1f}度。ほぼ垂直ですが、\n"
            "パートナー側にわずかに寄りかかっている傾向が見られます。\n\n"
            "【意識するイメージ】\n"
            "「自分の中にまっすぐな柱がある」ことを忘れずに。\n"
            "パートナーの手は「方向を示す道しるべ」であって、\n"
            "「もたれかかる壁」ではないと思ってみましょう。\n\n"
            "【おすすめの練習】\n"
            "サポートなしでポーズを3秒キープ → パートナーが手を添える、\n"
            "という順番で練習すると、自分の軸の感覚がつかめます。"
        )
    else:
        msg = (
            "【現状】\n"
            f"体幹が垂直から{angle:.1f}度傾いています。\n"
            "パートナーに頼りすぎているか、体幹の引き上げが足りない可能性があります。\n\n"
            "【意識するイメージ】\n"
            "「自分一人でも立てるけど、パートナーがいてくれるからもっと美しくなれる」\n"
            "── そんな気持ちで立ってみましょう。\n"
            "相手を信頼することと、自分の軸を保つことは両立できます。\n\n"
            "【おすすめの練習】\n"
            "まずはバーで一人でポーズを安定させてから、\n"
            "パートナーのサポートに移行しましょう。\n"
            "「サポートがなくなっても倒れない」自信を体に覚えさせることが大切です。"
        )

    advice_list.append(Advice(
        metric_name="体幹の垂直性",
        score=round(s, 1),
        level=level,
        message=msg,
        priority=2,
    ))

    # --- 3. サポート距離 ---
    s = scores["support_distance"]
    level = _score_to_level(s)
    dist = metrics.support_distance

    if level == "excellent":
        msg = (
            "【すばらしい！】\n"
            f"二人の距離感がとても自然で美しいです。（距離{dist:.3f}）\n"
            "お互いの呼吸が合った、心地よい「間（ま）」が生まれています。\n\n"
            "【さらに磨くなら】\n"
            "「見えない糸で二人が繋がっている」ような一体感を。\n"
            "距離が変わっても、その糸がピンと張ったまま保たれるイメージです。\n\n"
            "【おすすめの練習】\n"
            "プロムナードの最中も、二人の距離が変わらないことを意識して。\n"
            "鏡を見ながら、二人のシルエットが一つの美しい形を描いているか確認しましょう。"
        )
    elif level == "good":
        if dist < 0.15:
            detail = "少し近すぎる傾向があります。"
            image = (
                "「お互いの周りに透明なオーラがある」と想像してみましょう。\n"
                "そのオーラが重なりすぎないよう、ほんの少し空間を開けると\n"
                "お互いの動きに余裕が生まれます。"
            )
        else:
            detail = "やや距離が開きすぎる傾向があります。"
            image = (
                "「二人で一つのスポットライトの中にいる」とイメージしてみましょう。\n"
                "光の輪から出ないよう、やさしく寄り添う距離感を意識してみてください。"
            )
        msg = (
            "【現状】\n"
            f"サポート距離は{dist:.3f}。{detail}\n\n"
            "【意識するイメージ】\n"
            f"{image}\n\n"
            "【おすすめの練習】\n"
            "音楽なしで、ゆっくり歩きながらの簡単なプロムナード練習がおすすめです。\n"
            "二人の間隔が一定に保たれるよう、相手の呼吸と歩幅に合わせてみましょう。"
        )
    else:
        if dist < 0.10:
            problem = "距離が近すぎて、お互いの動きが窮屈になっています。"
        else:
            problem = "距離が離れすぎて、サポートが不安定になりがちです。"
        msg = (
            "【現状】\n"
            f"サポート距離は{dist:.3f}。{problem}\n\n"
            "【意識するイメージ】\n"
            "「二人で大きな風船を優しく挟んでいる」と想像してみてください。\n"
            "風船を潰さない、でも落とさない ── そんなやさしい距離感です。\n"
            "パートナーの肩の力を感じたら、二人とも深呼吸してリラックスしましょう。\n\n"
            "【おすすめの練習】\n"
            "まずは向かい合って、腕一本分の距離で静止ポーズから。\n"
            "その距離感を体に覚えさせてから、動きのある技に移っていきましょう。"
        )

    advice_list.append(Advice(
        metric_name="サポート距離",
        score=round(s, 1),
        level=level,
        message=msg,
        priority=3,
    ))

    advice_list.sort(key=lambda a: a.priority)
    return advice_list


def evaluate_frame_pair(
    landmarks_a: list, landmarks_b: list,
    persons_dict: list[dict] | None = None,
) -> AnalysisResult:
    """2人のランドマークからパ・ド・ドゥの評価を行う。

    Args:
        landmarks_a: Person 0 のランドマーク
        landmarks_b: Person 1 のランドマーク
        persons_dict: JSON変換済みの人物ランドマークリスト

    Returns:
        AnalysisResult (pose_type="pas_de_deux")
    """
    metrics_obj = analyze_pas_de_deux(landmarks_a, landmarks_b)
    scores = metrics_obj.scores
    overall = _compute_overall(scores)
    advice = generate_pdd_advice(metrics_obj)

    metrics_dict = {
        "shared_com_displacement": metrics_obj.shared_com_displacement,
        "com_x": metrics_obj.com_x,
        "com_y": metrics_obj.com_y,
        "within_base": metrics_obj.within_base,
        "trunk_verticality": metrics_obj.trunk_verticality,
        "support_distance": metrics_obj.support_distance,
        "supported_person_id": metrics_obj.supported_person_id,
    }

    pair_data = {
        "persons": persons_dict or [],
        "pair_metrics": metrics_dict,
        "pair_scores": {k: round(v, 1) for k, v in scores.items()},
    }

    return AnalysisResult(
        pose_type="pas_de_deux",
        overall_score=round(overall, 1),
        metrics=metrics_dict,
        scores={k: round(v, 1) for k, v in scores.items()},
        advice=advice,
        pair_data=pair_data,
    )


def evaluate_video_pair(
    frames_multi: list,
    frame_indices: list[int] | None = None,
    frame_timestamps: list[float] | None = None,
    video_duration_ms: float = 0.0,
) -> AnalysisResult:
    """複数フレームのパ・ド・ドゥ動画解析。

    Args:
        frames_multi: MultiFrameResult のリスト (各要素に .persons がある)
        frame_indices: フレームインデックスリスト
        frame_timestamps: タイムスタンプリスト
        video_duration_ms: 動画の総時間

    Returns:
        AnalysisResult (ベストフレーム基準)
    """
    if not frames_multi:
        return AnalysisResult(
            pose_type="pas_de_deux",
            overall_score=0.0,
            metrics={},
            scores={},
            advice=[Advice("入力エラー", 0, "needs_work",
                          "2人のポーズを検出できるフレームがありませんでした。", 1)],
        )

    _indices = frame_indices or [mf.frame_index for mf in frames_multi]
    _timestamps = frame_timestamps or [mf.timestamp_ms for mf in frames_multi]

    best_result = None
    best_score = -1.0
    best_idx = 0

    for i, mf in enumerate(frames_multi):
        persons = sorted(mf.persons, key=lambda p: p.person_id)
        if len(persons) < 2:
            continue
        persons_dict = [p.to_dict() for p in persons]
        result = evaluate_frame_pair(
            persons[0].landmarks, persons[1].landmarks,
            persons_dict=persons_dict,
        )
        if result.overall_score > best_score:
            best_score = result.overall_score
            best_result = result
            best_idx = i

    if best_result is None:
        return AnalysisResult(
            pose_type="pas_de_deux",
            overall_score=0.0,
            metrics={},
            scores={},
            advice=[Advice("入力エラー", 0, "needs_work",
                          "2人のポーズを検出できるフレームがありませんでした。", 1)],
        )

    best_result.best_frame_index = _indices[best_idx]
    best_result.best_frame_timestamp_ms = _timestamps[best_idx]
    best_result.total_frames_analyzed = len(frames_multi)
    best_result.video_duration_ms = video_duration_ms

    return best_result
