import { useState, useCallback } from "react";
import { API_BASE } from "../config";

export interface LandmarkPoint {
  index: number;
  x: number;
  y: number;
  z: number;
  visibility: number;
}

export interface FrameLandmarks {
  frame_index: number;
  timestamp_ms: number;
  landmarks: LandmarkPoint[];
}

export interface AdviceItem {
  metric_name: string;
  score: number;
  level: "excellent" | "good" | "needs_work";
  message: string;
  priority: number;
}

export interface AnalysisResponse {
  pose_type: string;
  overall_score: number;
  metrics: Record<string, number | string>;
  scores: Record<string, number>;
  advice: AdviceItem[];
  stability: { stability_score: number; sway_x: number; sway_y: number } | null;
  landmarks: FrameLandmarks[];
  best_frame_index: number;
  best_frame_timestamp_ms: number;
  total_frames_analyzed: number;
  video_duration_ms: number;
  best_frame_image?: string; // base64 JPEG (動画解析時のみ)
  rotation_data?: {
    rotation_count: number;
    avg_seconds_per_turn: number;
    rpm: number;
    rotation_start_ms: number;
    rotation_end_ms: number;
    rotation_duration_ms: number;
    peak_speed_rpm: number;
  };
}

export function useAnalysis() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [phase, setPhase] = useState<"idle" | "uploading" | "analyzing">("idle");

  const analyze = useCallback(async (file: File) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setUploadProgress(0);
    setPhase("uploading");

    const formData = new FormData();
    formData.append("file", file);

    try {
      // Renderスリープからの復帰を事前に行う（バックグラウンド）
      try {
        await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(60000) });
      } catch {
        // ヘルスチェック失敗でも続行
      }

      // XMLHttpRequest for upload progress tracking
      const data = await new Promise<AnalysisResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/api/analyze`);
        xhr.timeout = 300000; // 5分タイムアウト

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            setUploadProgress(Math.round((e.loaded / e.total) * 100));
          }
        };
        xhr.upload.onload = () => {
          setPhase("analyzing");
          setUploadProgress(100);
        };
        xhr.onload = () => {
          try {
            const body = JSON.parse(xhr.responseText);
            if (xhr.status >= 200 && xhr.status < 300) {
              resolve(body as AnalysisResponse);
            } else {
              reject(new Error(body.detail || `Error ${xhr.status}`));
            }
          } catch {
            reject(new Error(`Error ${xhr.status}`));
          }
        };
        xhr.onerror = () => reject(new Error("ネットワークエラー。サーバーが起動中の場合があります。少し待ってから再度お試しください。"));
        xhr.ontimeout = () => reject(new Error("タイムアウト。動画が長すぎる可能性があります。短い動画でお試しください。"));
        xhr.send(formData);
      });

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "解析に失敗しました");
    } finally {
      setLoading(false);
      setPhase("idle");
    }
  }, []);

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { result, loading, error, analyze, reset, uploadProgress, phase };
}
