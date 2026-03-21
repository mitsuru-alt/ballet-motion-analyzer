from __future__ import annotations

"""
Pirouette Analyzer - FastAPI Backend

ピルエット（回転）専門の軽量バレエ解析API。

エンドポイント:
  POST /api/analyze       画像/動画を解析
  POST /api/history       解析結果を履歴に保存
  GET  /api/history       履歴を取得
  DELETE /api/history/{id} 履歴レコードを削除
  GET  /health            ヘルスチェック
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
from typing import Any, Dict, List, Optional

from .pose_estimator import PoseEstimator, _save_video_tempfile
from .scoring_engine import (
    evaluate_frame, evaluate_video,
    Advice, AnalysisResult,
)
from .history_db import save_analysis, get_history, get_history_count, delete_record

app = FastAPI(title="Pirouette Analyzer", version="2.0.0")

# CORS
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

estimator = PoseEstimator()

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
    return resp


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """画像または動画を解析（ピルエット専門）"""
    content_type = file.content_type or ""
    original_filename = file.filename or ""

    # iOS content_type フォールバック
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
            frame_result = estimator.process_image(file_bytes)
            if frame_result is None:
                raise HTTPException(
                    422,
                    "ポーズを検出できませんでした。全身が映る画像を使用してください。",
                )
            analysis = evaluate_frame(frame_result.landmarks)
            analysis.landmarks_data = [frame_result.to_dict()]
            return _result_to_response(analysis)

        elif content_type in SUPPORTED_VIDEO_TYPES:
            # 一時ファイルを1回だけ作成
            tmp_path = _save_video_tempfile(file_bytes, original_filename)
            del file_bytes  # メモリ早期解放

            try:
                # シングルパス: ポーズ評価用(3fps) + 回転検出用(15fps) を同時取得
                scan = estimator.process_video_single_pass(tmp_path)

                if not scan.frame_results:
                    raise HTTPException(
                        422,
                        "動画からポーズを検出できませんでした。全身が映る動画を使用してください。",
                    )

                analysis = evaluate_video(
                    [fr.landmarks for fr in scan.frame_results],
                    frame_indices=[fr.frame_index for fr in scan.frame_results],
                    frame_timestamps=[fr.timestamp_ms for fr in scan.frame_results],
                    video_duration_ms=scan.video_duration_ms,
                    dense_frames=scan.dense_frames,
                    source_fps=scan.source_fps,
                )
                analysis.landmarks_data = [fr.to_dict() for fr in scan.frame_results]

                # ベストフレーム画像（スキャン中に保持した画像を使用 = 動画再オープン不要）
                best_frame_image_b64 = None
                best_jpg = estimator.extract_best_frame_image(
                    scan, analysis.best_frame_index, video_path=tmp_path,
                )
                if best_jpg:
                    best_frame_image_b64 = base64.b64encode(best_jpg).decode("ascii")

                # スキャン結果の画像メモリを解放
                scan.frame_images = None

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
            "ファイルが大きすぎて処理できません。短い動画でお試しください。",
        )
    except Exception as e:
        print(f"[analyze] Error: {e}")
        raise HTTPException(
            500,
            "動画の処理中にエラーが発生しました。別の形式（MP4）で再度お試しください。",
        )


# ============================================================
# 履歴 API
# ============================================================

class HistorySaveRequest(BaseModel):
    pose_type: str
    overall_score: float
    scores: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    advice: List[Dict[str, Any]] = []
    rotation_data: Optional[Dict[str, Any]] = None


@app.post("/api/history")
async def save_history(req: HistorySaveRequest):
    record_id = save_analysis(req.model_dump())
    return {"id": record_id, "status": "saved"}


@app.get("/api/history")
async def list_history(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    pose_type: Optional[str] = Query(None),
):
    records = get_history(limit=limit, offset=offset, pose_type=pose_type)
    total = get_history_count(pose_type=pose_type)
    return {"records": records, "total": total}


@app.delete("/api/history/{record_id}")
async def remove_history(record_id: int):
    if delete_record(record_id):
        return {"status": "deleted"}
    raise HTTPException(404, "Record not found")
