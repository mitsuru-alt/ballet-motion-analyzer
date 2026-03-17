import { useState, useCallback } from "react";
import { API_BASE } from "../config";

export interface HistoryRecord {
  id: number;
  created_at: string;
  pose_type: string;
  overall_score: number;
  scores: Record<string, number>;
  metrics: Record<string, number | string>;
  advice: Array<{
    metric_name: string;
    score: number;
    level: string;
    message: string;
    priority: number;
  }>;
  rotation_data: {
    rotation_count: number;
    avg_seconds_per_turn: number;
    rpm: number;
  } | null;
  pair_data: Record<string, unknown> | null;
}

export interface GrowthHighlight {
  icon: string;
  message: string;
  type: "improvement" | "stable" | "info";
}

const POSE_LABELS: Record<string, string> = {
  arabesque: "アラベスク",
  passe: "パッセ",
  pirouette: "ピルエット",
  pas_de_deux: "パ・ド・ドゥ",
};

const METRIC_LABELS: Record<string, string> = {
  leg_angle: "脚の挙上角度",
  back_line: "背中のライン",
  knee_extension: "膝の伸展",
  alignment: "アライメント",
  standing_knee: "軸脚の膝",
  passe_knee: "パッセ膝角度",
  toe_position: "つま先位置",
  hip_level: "骨盤の水平性",
  turnout: "アン・ドゥオール",
  releve_height: "ルルヴェの高さ",
  trunk_vertical: "体幹の垂直性",
  arm_position: "腕のポジション",
  passe_position: "パッセ脚の位置",
  shared_com: "重心安定性",
  trunk_angle: "体幹角度",
  support_distance: "サポート距離",
};

export function useHistory() {
  const [records, setRecords] = useState<HistoryRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);

  const fetchHistory = useCallback(async (poseType?: string) => {
    setLoading(true);
    try {
      const params = new URLSearchParams({ limit: "200" });
      if (poseType) params.set("pose_type", poseType);
      const res = await fetch(`${API_BASE}/api/history?${params}`);
      if (!res.ok) throw new Error("Failed to fetch history");
      const data = await res.json();
      setRecords(data.records);
      setTotal(data.total);
    } catch (e) {
      console.error("History fetch error:", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const saveToHistory = useCallback(
    async (analysisResult: {
      pose_type: string;
      overall_score: number;
      scores: Record<string, number>;
      metrics: Record<string, number | string>;
      advice: unknown[];
      rotation_data?: unknown;
      pair_data?: unknown;
    }) => {
      try {
        const res = await fetch(`${API_BASE}/api/history`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(analysisResult),
        });
        if (!res.ok) throw new Error("Failed to save");
        const data = await res.json();
        // 保存後に履歴をリフレッシュ
        await fetchHistory();
        return data.id;
      } catch (e) {
        console.error("History save error:", e);
        return null;
      }
    },
    [fetchHistory]
  );

  const deleteFromHistory = useCallback(
    async (id: number) => {
      try {
        const res = await fetch(`${API_BASE}/api/history/${id}`, { method: "DELETE" });
        if (!res.ok) throw new Error("Failed to delete");
        await fetchHistory();
      } catch (e) {
        console.error("History delete error:", e);
      }
    },
    [fetchHistory]
  );

  /** 成長ハイライトメッセージを生成 */
  const generateHighlights = useCallback((): GrowthHighlight[] => {
    if (records.length < 2) return [];

    const highlights: GrowthHighlight[] = [];
    // 直近のレコードと同じポーズタイプの過去レコードを比較
    const latest = records[0];

    // 同じポーズタイプの過去レコードを検索
    const sameTypeRecords = records.filter(
      (r) => r.pose_type === latest.pose_type
    );

    if (sameTypeRecords.length >= 2) {
      const prev = sameTypeRecords[1];
      const poseName = POSE_LABELS[latest.pose_type] || latest.pose_type;

      // 総合スコア比較
      const scoreDiff = latest.overall_score - prev.overall_score;
      if (scoreDiff > 0) {
        highlights.push({
          icon: "✨",
          message: `${poseName}の総合スコアが${scoreDiff.toFixed(0)}点アップしました！`,
          type: "improvement",
        });
      } else if (scoreDiff === 0) {
        highlights.push({
          icon: "💪",
          message: `${poseName}のスコアを安定してキープしています！`,
          type: "stable",
        });
      }

      // 各メトリクスの比較
      for (const [key, currentScore] of Object.entries(latest.scores)) {
        const prevScore = prev.scores[key];
        if (prevScore !== undefined) {
          const diff = currentScore - prevScore;
          const label = METRIC_LABELS[key] || key;
          if (diff >= 5) {
            highlights.push({
              icon: "🌟",
              message: `${label}が${diff.toFixed(0)}点向上しました！すばらしい成長です！`,
              type: "improvement",
            });
          }
        }
      }

      // 角度系メトリクスの比較
      for (const [key, currentVal] of Object.entries(latest.metrics)) {
        const prevVal = prev.metrics[key];
        if (typeof currentVal === "number" && typeof prevVal === "number") {
          const label = METRIC_LABELS[key] || key;
          if (
            key.includes("angle") ||
            key.includes("knee") ||
            key === "leg_angle"
          ) {
            const diff = currentVal - prevVal;
            if (Math.abs(diff) >= 3) {
              highlights.push({
                icon: diff > 0 ? "📐" : "📏",
                message: `${label}が前回より${Math.abs(diff).toFixed(1)}°${diff > 0 ? "大きく" : "改善されて"}なりました！`,
                type: "improvement",
              });
            }
          }
        }
      }
    }

    // 総解析回数のマイルストーン
    if (records.length === 5) {
      highlights.push({
        icon: "🎉",
        message: "5回の解析を達成！継続は力なりです！",
        type: "info",
      });
    } else if (records.length === 10) {
      highlights.push({
        icon: "🏆",
        message:
          "10回の解析達成！着実に成長の記録を積み重ねていますね！",
        type: "info",
      });
    } else if (records.length === 20) {
      highlights.push({
        icon: "👑",
        message: "20回達成！あなたの努力は本物です！",
        type: "info",
      });
    }

    return highlights.slice(0, 5); // 最大5件
  }, [records]);

  return {
    records,
    total,
    loading,
    fetchHistory,
    saveToHistory,
    deleteFromHistory,
    generateHighlights,
  };
}
