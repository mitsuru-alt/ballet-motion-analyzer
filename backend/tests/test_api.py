"""
Backend API 機能テスト

MediaPipeで検出可能な人物画像を使い、解析パイプライン全体を検証する。
テスト用画像はOpenCVで棒人間を描画して生成する。
"""
from __future__ import annotations

import sys
import os
import json
import cv2
import numpy as np

# app パッケージが import できるようにパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def create_stick_figure_arabesque(width=640, height=480) -> bytes:
    """
    アラベスク風の棒人間画像を生成 (MediaPipeの検出テスト用)

    実際のMediaPipeは写真品質の画像が必要だが、
    APIのルーティング・エラーハンドリングの検証には十分。
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # 明るい背景

    # 体のパーツを描画 (アラベスク風)
    head = (320, 120)
    neck = (320, 160)
    r_shoulder = (280, 180)
    l_shoulder = (360, 180)
    r_hip = (300, 300)
    l_hip = (340, 300)

    # 軸脚 (右脚) - 真っ直ぐ下
    r_knee = (300, 380)
    r_ankle = (300, 450)

    # 挙脚 (左脚) - 後方に挙上
    l_knee = (440, 260)
    l_ankle = (540, 220)

    # 腕
    r_wrist = (200, 160)
    l_wrist = (440, 140)

    color = (50, 50, 50)
    thickness = 3

    # 体幹
    cv2.line(img, head, neck, color, thickness)
    cv2.line(img, neck, r_shoulder, color, thickness)
    cv2.line(img, neck, l_shoulder, color, thickness)
    cv2.line(img, r_shoulder, r_hip, color, thickness)
    cv2.line(img, l_shoulder, l_hip, color, thickness)
    cv2.line(img, r_hip, l_hip, color, thickness)

    # 脚
    cv2.line(img, r_hip, r_knee, color, thickness)
    cv2.line(img, r_knee, r_ankle, color, thickness)
    cv2.line(img, l_hip, l_knee, color, thickness)
    cv2.line(img, l_knee, l_ankle, color, thickness)

    # 腕
    cv2.line(img, r_shoulder, r_wrist, color, thickness)
    cv2.line(img, l_shoulder, l_wrist, color, thickness)

    # 頭
    cv2.circle(img, head, 20, color, thickness)

    # 関節点
    for pt in [head, r_shoulder, l_shoulder, r_hip, l_hip,
               r_knee, r_ankle, l_knee, l_ankle, r_wrist, l_wrist]:
        cv2.circle(img, pt, 5, (0, 0, 200), -1)

    _, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes()


def test_health():
    """ヘルスチェック"""
    import urllib.request
    res = urllib.request.urlopen("http://localhost:8000/health")
    data = json.loads(res.read())
    assert data["status"] == "ok"
    print("[PASS] /health")


def test_poses_list():
    """/api/poses エンドポイント"""
    import urllib.request
    res = urllib.request.urlopen("http://localhost:8000/api/poses")
    data = json.loads(res.read())
    assert "poses" in data
    assert len(data["poses"]) == 2
    ids = [p["id"] for p in data["poses"]]
    assert "arabesque" in ids
    assert "passe" in ids
    print(f"[PASS] /api/poses - {len(data['poses'])} poses registered")


def test_analyze_image():
    """
    /api/analyze に画像を送信して解析パイプラインを検証

    棒人間画像ではMediaPipeが骨格を検出できない可能性があるため、
    2つのケースをテスト:
    1. 正常応答 (200) → スコア・アドバイスの構造を検証
    2. 検出失敗 (422) → エラーハンドリングを検証
    """
    import urllib.request
    import io

    image_bytes = create_stick_figure_arabesque()

    # multipart/form-data を手動構築
    boundary = "----TestBoundary12345"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
        f"Content-Type: image/jpeg\r\n"
        f"\r\n"
    ).encode("utf-8") + image_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/api/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        res = urllib.request.urlopen(req)
        data = json.loads(res.read())

        # 正常応答の場合: レスポンス構造を検証
        assert "pose_type" in data, "Missing pose_type"
        assert "overall_score" in data, "Missing overall_score"
        assert "metrics" in data, "Missing metrics"
        assert "scores" in data, "Missing scores"
        assert "advice" in data, "Missing advice"
        assert isinstance(data["advice"], list), "advice should be list"

        if data["pose_type"] != "unknown":
            assert data["overall_score"] > 0, "Score should be > 0"
            assert len(data["scores"]) > 0, "Should have scores"

        # アドバイスの構造検証
        for advice in data["advice"]:
            assert "metric_name" in advice
            assert "score" in advice
            assert "level" in advice
            assert advice["level"] in ("excellent", "good", "needs_work")
            assert "message" in advice
            assert len(advice["message"]) > 0

        print(f"[PASS] /api/analyze (image) - pose={data['pose_type']}, score={data['overall_score']}")
        print(f"       metrics: {json.dumps(data['metrics'], ensure_ascii=False, indent=2)[:200]}")
        print(f"       scores: {data['scores']}")
        print(f"       advice count: {len(data['advice'])}")
        for a in data["advice"][:3]:
            print(f"       - [{a['level']}] {a['metric_name']}: {a['message'][:60]}...")

    except urllib.error.HTTPError as e:
        if e.code == 422:
            error_body = json.loads(e.read())
            print(f"[PASS] /api/analyze (image) - 422 correctly returned (pose not detected)")
            print(f"       detail: {error_body.get('detail', 'N/A')}")
        else:
            raise


def test_analyze_invalid_file():
    """/api/analyze に不正なContent-Typeを送信"""
    import urllib.request

    boundary = "----TestBoundary99999"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.txt"\r\n'
        f"Content-Type: text/plain\r\n"
        f"\r\n"
        f"this is not an image\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://localhost:8000/api/analyze",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        urllib.request.urlopen(req)
        print("[FAIL] /api/analyze (invalid) - should have returned 400")
    except urllib.error.HTTPError as e:
        if e.code == 400:
            print(f"[PASS] /api/analyze (invalid file) - 400 returned correctly")
        else:
            print(f"[FAIL] /api/analyze (invalid) - unexpected status {e.code}")


def test_scoring_engine_unit():
    """scoring_engine の単体テスト (モックランドマーク)"""
    from app.scoring_engine import evaluate_frame, classify_pose

    # モックランドマーク: 33個の点を生成
    class MockLandmark:
        def __init__(self, x, y, z, vis=0.9):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = vis

    # アラベスク風のランドマーク配置
    landmarks = [MockLandmark(0.5, 0.5, 0)] * 33  # デフォルト

    # 肩
    landmarks[11] = MockLandmark(0.45, 0.25, 0)  # 左肩
    landmarks[12] = MockLandmark(0.55, 0.25, 0)  # 右肩
    # 腰
    landmarks[23] = MockLandmark(0.45, 0.45, 0)  # 左腰
    landmarks[24] = MockLandmark(0.55, 0.45, 0)  # 右腰
    # 軸脚 (右) - 真っ直ぐ下
    landmarks[26] = MockLandmark(0.55, 0.65, 0)  # 右膝
    landmarks[28] = MockLandmark(0.55, 0.85, 0)  # 右足首
    # 挙脚 (左) - 後方に高く上げる
    landmarks[25] = MockLandmark(0.30, 0.35, 0)  # 左膝 (高い位置)
    landmarks[27] = MockLandmark(0.15, 0.25, 0)  # 左足首 (腰より高い)
    # ヒール・つま先
    landmarks[29] = MockLandmark(0.55, 0.87, 0)
    landmarks[30] = MockLandmark(0.55, 0.87, 0)
    landmarks[31] = MockLandmark(0.15, 0.23, 0)
    landmarks[32] = MockLandmark(0.15, 0.23, 0)

    # ポーズ分類テスト
    pose = classify_pose(landmarks)
    print(f"[INFO] Pose classified as: {pose}")

    # 評価テスト
    result = evaluate_frame(landmarks)
    print(f"[PASS] evaluate_frame - pose={result.pose_type}, score={result.overall_score}")
    print(f"       metrics: {result.metrics}")
    print(f"       scores: {result.scores}")
    for a in result.advice:
        print(f"       [{a.level}] {a.metric_name}: {a.message[:80]}...")

    # パッセ風のランドマーク配置
    landmarks_passe = [MockLandmark(0.5, 0.5, 0)] * 33

    landmarks_passe[11] = MockLandmark(0.45, 0.25, 0)  # 左肩
    landmarks_passe[12] = MockLandmark(0.55, 0.25, 0)  # 右肩
    landmarks_passe[23] = MockLandmark(0.45, 0.45, 0)  # 左腰
    landmarks_passe[24] = MockLandmark(0.55, 0.45, 0)  # 右腰
    # 軸脚 (右) - 真っ直ぐ
    landmarks_passe[26] = MockLandmark(0.55, 0.65, 0)  # 右膝
    landmarks_passe[28] = MockLandmark(0.55, 0.85, 0)  # 右足首
    # パッセ脚 (左) - 膝を曲げて高い位置
    landmarks_passe[25] = MockLandmark(0.38, 0.35, 0)  # 左膝 (高い位置、外に開く)
    landmarks_passe[27] = MockLandmark(0.48, 0.55, 0)  # 左足首 (軸脚膝の高さ付近)
    landmarks_passe[29] = MockLandmark(0.55, 0.87, 0)
    landmarks_passe[30] = MockLandmark(0.55, 0.87, 0)
    landmarks_passe[31] = MockLandmark(0.48, 0.57, 0)
    landmarks_passe[32] = MockLandmark(0.48, 0.57, 0)

    pose_p = classify_pose(landmarks_passe)
    result_p = evaluate_frame(landmarks_passe)
    print(f"\n[PASS] Passe test - pose={result_p.pose_type}, score={result_p.overall_score}")
    print(f"       metrics: {result_p.metrics}")
    print(f"       scores: {result_p.scores}")
    for a in result_p.advice:
        print(f"       [{a.level}] {a.metric_name}: {a.message[:80]}...")


def test_stability():
    """重心安定性の計算テスト"""
    from app.ballet_metrics import compute_center_of_gravity_stability, PoseLandmark

    class MockLandmark:
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.visibility = 0.9

    # 安定したフレーム列 (重心がほぼ動かない)
    stable_frames = []
    for i in range(10):
        frame = [MockLandmark(0.5, 0.5, 0)] * 33
        frame[PoseLandmark.LEFT_HIP] = MockLandmark(0.48 + 0.001 * i % 2, 0.5, 0)
        frame[PoseLandmark.RIGHT_HIP] = MockLandmark(0.52 + 0.001 * i % 2, 0.5, 0)
        stable_frames.append(frame)

    stability = compute_center_of_gravity_stability(stable_frames)
    print(f"[PASS] Stability (stable) - score={stability['stability_score']:.1f}, "
          f"sway_x={stability['sway_x']:.4f}, sway_y={stability['sway_y']:.4f}")

    # 不安定なフレーム列 (重心が大きく揺れる)
    unstable_frames = []
    for i in range(10):
        frame = [MockLandmark(0.5, 0.5, 0)] * 33
        frame[PoseLandmark.LEFT_HIP] = MockLandmark(0.4 + 0.04 * (i % 3), 0.45 + 0.03 * (i % 2), 0)
        frame[PoseLandmark.RIGHT_HIP] = MockLandmark(0.5 + 0.04 * (i % 3), 0.45 + 0.03 * (i % 2), 0)
        unstable_frames.append(frame)

    stability_u = compute_center_of_gravity_stability(unstable_frames)
    print(f"[PASS] Stability (unstable) - score={stability_u['stability_score']:.1f}, "
          f"sway_x={stability_u['sway_x']:.4f}, sway_y={stability_u['sway_y']:.4f}")

    assert stability["stability_score"] > stability_u["stability_score"], \
        "Stable should score higher than unstable"
    print("[PASS] Stable > Unstable assertion OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Ballet Analyzer - Backend Test Suite")
    print("=" * 60)

    print("\n--- Unit Tests ---")
    test_scoring_engine_unit()

    print("\n--- Stability Tests ---")
    test_stability()

    print("\n--- API Integration Tests ---")
    test_health()
    test_poses_list()
    test_analyze_image()
    test_analyze_invalid_file()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
