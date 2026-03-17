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

export interface PersonLandmarks {
  person_id: number;
  landmarks: LandmarkPoint[];
}

export interface PairMetrics {
  shared_com_displacement: number;
  com_x: number;
  com_y: number;
  within_base: boolean;
  trunk_verticality: number;
  support_distance: number;
  supported_person_id: number;
}

export interface PairData {
  persons: PersonLandmarks[];
  pair_metrics: PairMetrics;
  pair_scores: Record<string, number>;
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
  pair_data?: PairData; // パ・ド・ドゥ解析データ（2人検出時のみ）
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
      // XMLHttpRequest for upload progress tracking
      const data = await new Promise<AnalysisResponse>((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", `${API_BASE}/api/analyze`);

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
        xhr.onerror = () => reject(new Error("ネットワークエラー"));
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
