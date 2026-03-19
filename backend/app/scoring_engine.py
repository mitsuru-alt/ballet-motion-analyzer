from __future__ import annotations

"""
Pirouette Scoring Engine - ピルエット専門の評価・アドバイス生成

■ ピルエットの評価指標:
1. 軸脚の膝伸展 - 175-180度が理想
2. ルルヴェの高さ - かかとが十分に上がっているか
3. 体幹の垂直性 - 肩中心-腰中心の鉛直からの偏差
4. 腕のポジション - ひじの開きと手首の近さ
5. 骨盤の水平性 - 左右の腰の高さの差
6. パッセ脚の膝角度 - コンパクトに折りたたまれているか
"""

from dataclasses import dataclass, field
import numpy as np

from .ballet_metrics import (
    PoseLandmark,
    compute_angle_3points,
    compute_center_of_gravity_stability,
    compute_rotation_analysis,
    get_landmark_array,
)


# ============================================================
# ピルエット解析
# ============================================================

@dataclass
class PirouetteMetrics:
    """ピルエット解析結果"""
    standing_knee_angle: float
    releve_height: float
    vertical_axis_angle: float
    arm_position_score_raw: float
    pelvic_tilt: float
    working_knee_angle: float
    raised_side: str

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
        diff = abs(self.releve_height)
        if diff <= 0.01:
            return 100
        if diff >= 0.06:
            return 20
        return 20 + (0.06 - diff) / 0.05 * 80

    def _score_vertical_axis(self) -> float:
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

    standing_knee = compute_angle_3points(hip_s, knee_s, ankle_s)
    releve_height = float(heel_s[1] - foot_index_s[1])

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

    left_elbow = get_landmark_array(landmarks[PoseLandmark.LEFT_ELBOW])
    right_elbow = get_landmark_array(landmarks[PoseLandmark.RIGHT_ELBOW])
    left_wrist = get_landmark_array(landmarks[PoseLandmark.LEFT_WRIST])
    right_wrist = get_landmark_array(landmarks[PoseLandmark.RIGHT_WRIST])

    elbow_spread = abs(left_elbow[0] - right_elbow[0])
    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    spread_ratio = elbow_spread / max(shoulder_width, 0.01)
    spread_score = min(1.0, spread_ratio)

    wrist_dist = float(np.linalg.norm(left_wrist - right_wrist))
    wrist_ratio = wrist_dist / max(shoulder_width, 0.01)
    wrist_score = 1.0 if wrist_ratio <= 1.0 else max(0.0, 1.0 - (wrist_ratio - 1.0) * 0.5)
    arm_position_raw = 0.5 * spread_score + 0.5 * wrist_score

    pelvic_tilt = float(left_hip[1] - right_hip[1])
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
# 共通データ構造
# ============================================================

@dataclass
class Advice:
    """改善アドバイス1項目"""
    metric_name: str
    score: float
    level: str
    message: str
    priority: int


@dataclass
class AnalysisResult:
    """総合解析結果"""
    pose_type: str
    overall_score: float
    metrics: dict
    scores: dict
    advice: list[Advice] = field(default_factory=list)
    stability: dict | None = None
    landmarks_data: list[dict] = field(default_factory=list)
    best_frame_index: int = 0
    best_frame_timestamp_ms: float = 0.0
    total_frames_analyzed: int = 1
    video_duration_ms: float = 0.0
    rotation_data: dict | None = None


def _score_to_level(score: float) -> str:
    if score >= 90:
        return "excellent"
    elif score >= 70:
        return "good"
    return "needs_work"


def _compute_overall(scores: dict) -> float:
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


# ============================================================
# アドバイス生成
# ============================================================

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
            "回転中もひざが緩まない強さを意識しましょう。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセで16カウントキープ。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"軸足のひざは{angle:.0f}度。まっすぐ（{target:.0f}度）まであと{gap:.0f}度です！\n\n"
            "【意識するイメージ】\n"
            "「コマの軸のようにまっすぐ1本」── "
            "プリエから立ち上がる瞬間に、一気にひざを伸ばしきるのがポイントです。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、プリエ→ルルヴェ・パッセを繰り返しましょう。"
        )
    else:
        msg = (
            "【現状】\n"
            f"軸足のひざが曲がっています。（現在{angle:.0f}度 / 目標{target:.0f}度）\n\n"
            "【意識するイメージ】\n"
            "「コマの軸」を思い浮かべてください。"
            "おなかを上に引き上げながら、まっすぐ1本の軸を作る意識を。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、プリエからルルヴェの上げ下げを。"
        )
    advice_list.append(Advice("軸足のおひざ", s, _score_to_level(s), msg, 1))

    # ── ルルヴェの高さ ──
    s = scores["releve_height"]
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "ルルヴェの高さが十分！かかとがしっかり上がり、足の甲も美しく伸びています。\n\n"
            "【さらに磨くなら】\n"
            "「天井から糸で引き上げられている」イメージで体全体で上に伸びる感覚を。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセのキープ時間をどんどん延ばしていきましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "ルルヴェの高さはまずまず。もう少し欲しいところです。\n\n"
            "【意識するイメージ】\n"
            "「天井から糸で引き上げられている」ように、体全体を上に引き上げる意識を。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、8カウントでルルヴェをキープする練習を。"
        )
    else:
        msg = (
            "【現状】\n"
            "ルルヴェの高さがまだ足りません。かかとが上がりきっていません。\n\n"
            "【意識するイメージ】\n"
            "「天井から糸で引き上げられる」ように、足だけでなく体全体で上に伸びていく感覚を。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、ルルヴェの上げ下げをたくさん繰り返しましょう。"
        )
    advice_list.append(Advice("つま先立ちの高さ", s, _score_to_level(s), msg, 2))

    # ── 体の垂直性 ──
    s = scores["vertical_axis"]
    va = metrics.vertical_axis_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"体の傾き{va:.0f}度 ── ほぼ完璧にまっすぐ！美しい回転軸です。\n\n"
            "【さらに磨くなら】\n"
            "「体全体が1本の美しい柱になる」イメージで頭のてっぺんから天井へ伸び続けて。\n\n"
            "【おすすめの練習】\n"
            "スポットを決めて実際にピルエットを回りましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"体の傾きは{va:.0f}度 ── あと少しでまっすぐです！\n\n"
            "【意識するイメージ】\n"
            "「体全体が1本の柱になる」イメージ。おなかを引き込んで天井に向かって伸びて。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、ルルヴェ・パッセのまままっすぐ立つ練習を。"
        )
    else:
        msg = (
            "【現状】\n"
            f"体が{va:.0f}度傾いています。この状態では回転のコントロールが効きません。\n\n"
            "【意識するイメージ】\n"
            "「体全体が1本の柱」── 傾いた柱は倒れます。まずまっすぐ立つことを完璧に。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、ルルヴェ・パッセでまっすぐ立つ練習から始めましょう。"
        )
    advice_list.append(Advice("体のまっすぐさ", s, _score_to_level(s), msg, 1))

    # ── アームス ──
    s = scores["arm_position"]
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "アームスがとても美しい！丸みのある綺麗な形です。\n\n"
            "【さらに磨くなら】\n"
            "「大きなボールをそっと抱える」イメージで指先までやわらかさを意識して。\n\n"
            "【おすすめの練習】\n"
            "ポール・ド・ブラをゆっくり丁寧に練習しましょう。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "アームスはまずまず。もう少し磨けます！\n\n"
            "【意識するイメージ】\n"
            "「大きなボールをそっと抱える」イメージで肩を下げてひじを前にキープ。\n\n"
            "【おすすめの練習】\n"
            "鏡の前でアン・ナヴァンの形を取り、5秒キープを繰り返しましょう。"
        )
    else:
        msg = (
            "【現状】\n"
            "アームスの形が崩れています。ひじが下がっていませんか？\n\n"
            "【意識するイメージ】\n"
            "「大きなボールをそっと抱える」── 力ではなく「形を保つ」意識です。\n\n"
            "【おすすめの練習】\n"
            "ポール・ド・ブラをゆっくり繰り返し、腕の通り道を体で覚えましょう。"
        )
    advice_list.append(Advice("うでのかたち", s, _score_to_level(s), msg, 3))

    # ── 腰の水平 ──
    s = scores["pelvic_level"]
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            "腰がまっすぐ水平 ── 回転しても軸がぶれない素晴らしい安定感！\n\n"
            "【さらに磨くなら】\n"
            "「腰にフラフープを水平に回す」イメージで回転中も水平を保って。\n\n"
            "【おすすめの練習】\n"
            "ピルエットを回りながら腰の水平を保つ練習を。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            "腰がほんの少し傾いています。回転のぶれにつながるので注意！\n\n"
            "【意識するイメージ】\n"
            "「腰にフラフープを水平に回す」イメージ。おなかに力を入れて水平キープ。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまり、パッセで鏡で左右の腰の高さが揃っているか確認。"
        )
    else:
        msg = (
            "【現状】\n"
            "腰が傾いて回転が不安定になっています。\n\n"
            "【意識するイメージ】\n"
            "「腰にフラフープを水平に」── 傾いたら落ちます。腰の水平を最優先に。\n\n"
            "【おすすめの練習】\n"
            "バーに両手でつかまり、腰をまっすぐに保ったままパッセの練習を。"
        )
    advice_list.append(Advice("腰の水平", s, _score_to_level(s), msg, 2))

    # ── 上げた足 ──
    s = scores["working_leg"]
    wk = metrics.working_knee_angle
    if s >= 90:
        msg = (
            "【すばらしい！】\n"
            f"上げた足のひざ{wk:.0f}度 ── コンパクトで美しいルティレの形！\n\n"
            "【さらに磨くなら】\n"
            "「三角形をキュッと折りたたむ」ように更にコンパクトに。回転が速くなります。\n\n"
            "【おすすめの練習】\n"
            "ルルヴェ・パッセで足の形を鏡チェック。回転中も崩さない力をつけて。"
        )
    elif s >= 70:
        msg = (
            "【現状】\n"
            f"上げた足のひざは{wk:.0f}度。あと少しで理想の形です！\n\n"
            "【意識するイメージ】\n"
            "「三角形をキュッとコンパクトに」── つま先を軸足のひざの横にしっかりつけて。\n\n"
            "【おすすめの練習】\n"
            "バーでルルヴェ・パッセの形を取り、正しいポジションでキープする練習を。"
        )
    else:
        msg = (
            "【現状】\n"
            f"上げた足のひざが{wk:.0f}度。回転がぶれやすい状態です。\n\n"
            "【意識するイメージ】\n"
            "「三角形をキュッと折りたたむ」こと最優先。小さくまとめて回転を安定させて。\n\n"
            "【おすすめの練習】\n"
            "バーにつかまって、つま先で軸足の内側を滑らせながら上に持ち上げる練習を。"
        )
    advice_list.append(Advice("上げた足のかたち", s, _score_to_level(s), msg, 3))

    # すべて高得点
    if all(v >= 90 for v in scores.values()):
        advice_list.append(Advice(
            "総合評価", min(scores.values()), "excellent",
            "【すばらしい！】\n"
            "すべて高水準です！美しいピルエットのフォームが完成しています。\n\n"
            "【さらに磨くなら】\n"
            "ダブル・ピルエットやアン・ドゥオールにも挑戦してみましょう！",
            5,
        ))

    advice_list.sort(key=lambda a: a.priority)
    return advice_list


# ============================================================
# 評価エントリポイント
# ============================================================

def evaluate_frame(landmarks: list) -> AnalysisResult:
    """1フレームのランドマークからピルエット評価"""
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

    return AnalysisResult(
        pose_type="pirouette",
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
    """複数フレームの動画解析"""
    if not frames_landmarks:
        return AnalysisResult(
            pose_type="pirouette",
            overall_score=0.0,
            metrics={},
            scores={},
            advice=[Advice("入力エラー", 0, "needs_work", "ポーズを検出できるフレームがありませんでした。", 1)],
        )

    _indices = frame_indices or list(range(len(frames_landmarks)))
    _timestamps = frame_timestamps or [0.0] * len(frames_landmarks)

    best_result = None
    best_score = -1.0
    best_idx = 0

    for i, landmarks in enumerate(frames_landmarks):
        result = evaluate_frame(landmarks)
        if result.overall_score > best_score:
            best_score = result.overall_score
            best_result = result
            best_idx = i

    best_result.best_frame_index = _indices[best_idx]
    best_result.best_frame_timestamp_ms = _timestamps[best_idx]
    best_result.total_frames_analyzed = len(frames_landmarks)
    best_result.video_duration_ms = video_duration_ms

    # 重心安定性
    stability = compute_center_of_gravity_stability(frames_landmarks)
    best_result.stability = stability

    if stability["stability_score"] < 70:
        best_result.advice.append(Advice(
            "重心の安定性",
            stability["stability_score"],
            _score_to_level(stability["stability_score"]),
            f"重心の安定性スコアは {stability['stability_score']:.0f}/100 です。"
            "ドゥミ・プリエからゆっくりポーズに入り、体幹の引き上げを意識すると安定感が増します。",
            0,
        ))
        best_result.advice.sort(key=lambda a: a.priority)

    # 回転解析
    if dense_frames:
        rotation_data = compute_rotation_analysis(
            dense_frames, source_fps, video_duration_ms
        )
        best_result.rotation_data = rotation_data

    return best_result
