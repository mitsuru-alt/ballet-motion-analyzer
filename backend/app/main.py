from __future__ import annotations

"""
Ballet Motion Analyzer - FastAPI Backend

エンドポイント:
  POST /api/analyze       画像/動画ファイルを解析
  POST /api/analyze/image 画像のみ解析 (軽量)
  GET  /api/poses         対応ポーズ一覧
  POST /api/history       解析結果を履歴に保存
  GET  /api/history       保存された履歴を取得
  DELETE /api/history/{id} 履歴レコードを削除
  GET  /health            ヘルスチェック
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import base64
import io
import os
from typing import Any, Dict, List, Optional

from .pose_estimator import PoseEstimator, _save_video_tempfile
from .scoring_engine import (
    evaluate_frame, evaluate_video, evaluate_frame_pair, evaluate_video_pair,
    Advice, AnalysisResult,
)
from .history_db import save_analysis, get_history, get_history_count, delete_record

app = FastAPI(title="Ballet Motion Analyzer", version="0.1.0")

# CORS: ローカル開発 + Vercel本番ドメイン
_CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _CORS_ORIGINS],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバルインスタンス (model_complexity=1 で精度と速度のバランスを取る)
estimator = PoseEstimator(model_complexity=1, sample_fps=3)

SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
SUPPORTED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"}


def _advice_to_dict(advice: Advice) -> dict:
    return {
        "metric_name": advice.metric_name,
        "score": round(advice.score, 1),
        "level": advice.level,
        "message": advice.message,
        "priority": advice.priority,
    }


def _result_to_response(result: AnalysisResult, best_frame_image_b64: str | None = None) -> dict:
    resp = {
        "pose_type": result.pose_type,
        "overall_score": result.overall_score,
        "metrics": result.metrics,
        "scores": {k: round(v, 1) for k, v in result.scores.items()},
        "advice": [_advice_to_dict(a) for a in result.advice],
        "stability": result.stability,
        "landmarks": result.landmarks_data,
        "best_frame_index": result.best_frame_index,
        "best_frame_timestamp_ms": result.best_frame_timestamp_ms,
        "total_frames_analyzed": result.total_frames_analyzed,
        "video_duration_ms": result.video_duration_ms,
    }
    if best_frame_image_b64:
        resp["best_frame_image"] = best_frame_image_b64
    if result.rotation_data:
        resp["rotation_data"] = result.rotation_data
    if result.pair_data:
        resp["pair_data"] = result.pair_data
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/poses")
def list_poses():
    """対応しているバレエポーズの一覧"""
    return {
        "poses": [
            {
                "id": "arabesque",
                "name": "アラベスク",
                "description": "片脚を後方に高く上げ、上体を前傾させるポーズ",
                "metrics": [
                    "脚の挙上角度",
                    "背中のライン",
                    "膝の伸展",
                    "肩-腰-脚アライメント",
                ],
            },
            {
                "id": "passe",
                "name": "パッセ (ルティレ)",
                "description": "片脚の膝を曲げ、つま先を軸脚の膝に添えるポーズ",
                "metrics": [
                    "軸脚の膝伸展",
                    "パッセ脚の膝角度",
                    "つま先のポジション",
                    "骨盤の水平性",
                    "アン・ドゥオール",
                ],
            },
            {
                "id": "pirouette",
                "name": "ピルエット",
                "description": "片脚ルルヴェで回転するポーズ。パッセ脚＋ルルヴェ＋垂直体幹が特徴",
                "metrics": [
                    "軸脚の膝伸展",
                    "ルルヴェの高さ",
                    "体幹の垂直性",
                    "腕のポジション",
                    "骨盤の水平性",
                    "パッセ脚の位置",
                ],
            },
            {
                "id": "pas_de_deux",
                "name": "パ・ド・ドゥ",
                "description": "2人のダンサーによるペアダンス。サポートとバランスの調和を評価",
                "metrics": [
                    "共有重心の安定性",
                    "体幹の垂直性",
                    "サポート距離",
                ],
            },
        ]
    }


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    """
    静止画を解析し、ポーズの評価を返す

    - JPEG/PNG/WebP をサポート
    - レスポンスにランドマーク座標を含む (フロントエンドでのオーバーレイ描画用)
    """
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")

    image_bytes = await file.read()
    frame_result = estimator.process_image(image_bytes)

    if frame_result is None:
        raise HTTPException(
            422,
            "ポーズを検出できませんでした。全身が映る画像を使用してください。",
        )

    # 評価
    analysis = evaluate_frame(frame_result.landmarks)
    analysis.landmarks_data = [frame_result.to_dict()]

    return _result_to_response(analysis)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    画像または動画を解析

    Content-Type に応じて画像/動画を自動判別する。
    動画の場合はサンプリングFPSで間引き処理する。
    """
    content_type = file.content_type or ""
    original_filename = file.filename or ""

    # iOS は content_type が空や汎用型のことがあるので拡張子でも判定
    if content_type == "application/octet-stream" or not content_type:
        ext = os.path.splitext(original_filename)[1].lower()
        ext_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".webp": "image/webp", ".mp4": "video/mp4", ".mov": "video/quicktime",
            ".avi": "video/x-msvideo", ".webm": "video/webm", ".m4v": "video/mp4",
        }
        content_type = ext_map.get(ext, content_type)

    file_bytes = await file.read()

    try:
        if content_type in SUPPORTED_IMAGE_TYPES:
            # まず2人検出を試行（パ・ド・ドゥ判定）
            multi_result = estimator.process_image_multi(file_bytes)
            if multi_result is not None and len(multi_result.persons) >= 2:
                persons = sorted(multi_result.persons, key=lambda p: p.person_id)
                persons_dict = [p.to_dict() for p in persons]
                analysis = evaluate_frame_pair(
                    persons[0].landmarks, persons[1].landmarks,
                    persons_dict=persons_dict,
                )
                analysis.landmarks_data = [multi_result.to_dict()]
            else:
                # 1人の場合 → 既存の単一人物パス
                frame_result = estimator.process_image(file_bytes)
                if frame_result is None:
                    raise HTTPException(
                        422,
                        "ポーズを検出できませんでした。全身が映る画像を使用してください。",
                    )
                analysis = evaluate_frame(frame_result.landmarks)
                analysis.landmarks_data = [frame_result.to_dict()]

        elif content_type in SUPPORTED_VIDEO_TYPES:
            # 一時ファイルを1回だけ作り、全処理で使い回す（メモリ節約）
            tmp_path = _save_video_tempfile(file_bytes, original_filename)
            del file_bytes  # メモリ早期解放

            try:
                # まず2人検出を試行
                multi_results = estimator.process_video_multi_from_path(tmp_path)

                if len(multi_results) >= 3:
                    video_duration_ms = estimator.get_video_duration_ms_from_path(tmp_path)
                    analysis = evaluate_video_pair(
                        multi_results,
                        frame_indices=[mf.frame_index for mf in multi_results],
                        frame_timestamps=[mf.timestamp_ms for mf in multi_results],
                        video_duration_ms=video_duration_ms,
                    )
                    analysis.landmarks_data = [mf.to_dict() for mf in multi_results]

                    best_frame_image_b64 = None
                    best_frame_jpg = estimator.extract_frame_image_from_path(
                        tmp_path, analysis.best_frame_index,
                    )
                    if best_frame_jpg:
                        best_frame_image_b64 = base64.b64encode(best_frame_jpg).decode("ascii")

                    return _result_to_response(analysis, best_frame_image_b64=best_frame_image_b64)

                # 2人検出できなかった → 単一人物パス
                frame_results = estimator.process_video_from_path(tmp_path)
                if not frame_results:
                    raise HTTPException(
                        422,
                        "動画からポーズを検出できませんでした。全身が映る動画を使用してください。",
                    )

                all_landmarks = [fr.landmarks for fr in frame_results]
                frame_indices = [fr.frame_index for fr in frame_results]
                frame_timestamps = [fr.timestamp_ms for fr in frame_results]
                video_duration_ms = estimator.get_video_duration_ms_from_path(tmp_path)

                # 回転検出用の高頻度サンプリング
                dense_frames, source_fps = estimator.process_video_dense_from_path(tmp_path)

                analysis = evaluate_video(
                    all_landmarks,
                    frame_indices=frame_indices,
                    frame_timestamps=frame_timestamps,
                    video_duration_ms=video_duration_ms,
                    dense_frames=dense_frames,
                    source_fps=source_fps,
                )
                analysis.landmarks_data = [fr.to_dict() for fr in frame_results]

                best_frame_image_b64 = None
                best_frame_jpg = estimator.extract_frame_image_from_path(
                    tmp_path, analysis.best_frame_index,
                )
                if best_frame_jpg:
                    best_frame_image_b64 = base64.b64encode(best_frame_jpg).decode("ascii")

                return _result_to_response(analysis, best_frame_image_b64=best_frame_image_b64)
            finally:
                os.unlink(tmp_path)

        else:
            raise HTTPException(400, f"Unsupported file type: {content_type}")

    except HTTPException:
        raise
    except MemoryError:
        raise HTTPException(
            413,
            "ファイルが大きすぎて処理できません。shorter/低解像度の動画をお試しください。",
        )
    except Exception as e:
        print(f"[analyze] Error processing file: {e}")
        raise HTTPException(
            500,
            "動画の形式がサポートされていないか、処理中にエラーが発生しました。"
            "別の形式（MP4）で再度お試しください。",
        )

    return _result_to_response(analysis)


class HistorySaveRequest(BaseModel):
    pose_type: str
    overall_score: float
    scores: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    advice: List[Dict[str, Any]] = []
    rotation_data: Optional[Dict[str, Any]] = None
    pair_data: Optional[Dict[str, Any]] = None


@app.post("/api/history")
async def save_history(req: HistorySaveRequest):
    """解析結果を履歴に保存"""
    record_id = save_analysis(req.model_dump())
    return {"id": record_id, "status": "saved"}


@app.get("/api/history")
async def list_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pose_type: Optional[str] = Query(None),
):
    """保存された履歴データを取得"""
    records = get_history(limit=limit, offset=offset, pose_type=pose_type)
    total = get_history_count(pose_type=pose_type)
    return {"records": records, "total": total}


@app.delete("/api/history/{record_id}")
async def remove_history(record_id: int):
    """履歴レコードを削除"""
    if delete_record(record_id):
        return {"status": "deleted"}
    raise HTTPException(404, "Record not found")


@app.post("/api/analyze/overlay")
async def analyze_with_overlay(file: UploadFile = File(...)):
    """
    静止画を解析し、骨格オーバーレイ付き画像をJPEGで返す

    フロントエンド側でCanvas描画しない場合のフォールバック用。
    """
    if file.content_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported image type: {file.content_type}")

    image_bytes = await file.read()
    frame_result = estimator.process_image(image_bytes)

    if frame_result is None:
        raise HTTPException(422, "ポーズを検出できませんでした。")

    overlay_bytes = estimator.draw_skeleton(image_bytes, frame_result)
    return StreamingResponse(io.BytesIO(overlay_bytes), media_type="image/jpeg")
