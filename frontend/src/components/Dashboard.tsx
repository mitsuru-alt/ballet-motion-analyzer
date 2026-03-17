import { useState, useMemo } from "react";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from "recharts";
import type { HistoryRecord, GrowthHighlight } from "../hooks/useHistory";

interface Props {
  records: HistoryRecord[];
  highlights: GrowthHighlight[];
  loading: boolean;
  onBack: () => void;
}

const POSE_LABELS: Record<string, string> = {
  arabesque: "アラベスク",
  passe: "パッセ",
  pirouette: "ピルエット",
  pas_de_deux: "パ・ド・ドゥ",
};

const SCORE_COLORS: Record<string, string> = {
  overall_score: "#6366f1",
  leg_angle: "#f59e0b",
  back_line: "#10b981",
  knee_extension: "#3b82f6",
  alignment: "#8b5cf6",
  standing_knee: "#ec4899",
  passe_knee: "#f97316",
  toe_position: "#14b8a6",
  hip_level: "#a855f7",
  turnout: "#06b6d4",
  releve_height: "#22c55e",
  trunk_vertical: "#0ea5e9",
  arm_position: "#e11d48",
  passe_position: "#84cc16",
  shared_com: "#10b981",
  trunk_angle: "#3b82f6",
  support_distance: "#f59e0b",
};

const METRIC_LABELS: Record<string, string> = {
  overall_score: "総合スコア",
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

function formatDate(iso: string): string {
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
}

function formatDateShort(iso: string): string {
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

export default function Dashboard({
  records,
  highlights,
  loading,
  onBack,
}: Props) {
  const [filterPose, setFilterPose] = useState<string>("all");
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([
    "overall_score",
  ]);
  const [backHover, setBackHover] = useState(false);

  // フィルタリングされたレコード
  const filtered = useMemo(() => {
    let r = [...records];
    if (filterPose !== "all") {
      r = r.filter((rec) => rec.pose_type === filterPose);
    }
    return r.reverse(); // 古い順（グラフ用）
  }, [records, filterPose]);

  // グラフ用データ
  const chartData = useMemo(() => {
    return filtered.map((rec) => {
      const point: Record<string, number | string> = {
        date: formatDateShort(rec.created_at),
        fullDate: formatDate(rec.created_at),
        overall_score: rec.overall_score,
      };
      for (const [key, val] of Object.entries(rec.scores)) {
        point[key] = val;
      }
      return point;
    });
  }, [filtered]);

  // 利用可能なスコアキー
  const availableKeys = useMemo(() => {
    const keys = new Set<string>(["overall_score"]);
    for (const rec of filtered) {
      for (const k of Object.keys(rec.scores)) {
        keys.add(k);
      }
    }
    return Array.from(keys);
  }, [filtered]);

  // ポーズタイプ一覧
  const poseTypes = useMemo(() => {
    const types = new Set<string>();
    for (const rec of records) {
      types.add(rec.pose_type);
    }
    return Array.from(types);
  }, [records]);

  const toggleMetric = (key: string) => {
    setSelectedMetrics((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]
    );
  };

  if (loading) {
    return (
      <div style={{ textAlign: "center", padding: 80 }}>
        <div
          style={{
            width: 40,
            height: 40,
            border: "3px solid rgba(99,102,241,0.1)",
            borderTopColor: "#6366f1",
            borderRadius: "50%",
            animation: "spin 0.8s linear infinite",
            margin: "0 auto 16px",
          }}
        />
        <p style={{ color: "#64748b", fontSize: 14 }}>
          履歴データを読み込み中...
        </p>
      </div>
    );
  }

  return (
    <div
      style={{
        maxWidth: 900,
        margin: "0 auto",
        animation: "fadeIn 0.4s ease",
      }}
    >
      {/* ヘッダー */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 24,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <button
            onClick={onBack}
            onMouseOver={() => setBackHover(true)}
            onMouseOut={() => setBackHover(false)}
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              border: `1px solid ${backHover ? "#a5b4fc" : "rgba(226,232,240,0.8)"}`,
              background: backHover ? "#f8f9ff" : "rgba(255,255,255,0.9)",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 16,
              color: backHover ? "#4338ca" : "#64748b",
              transition: "all 0.2s ease",
              backdropFilter: "blur(8px)",
            }}
          >
            ←
          </button>
          <div>
            <h2
              style={{
                fontSize: 22,
                fontWeight: 800,
                color: "#0f172a",
                letterSpacing: "-0.02em",
                margin: 0,
              }}
            >
              成長記録
            </h2>
            <p style={{ fontSize: 12, color: "#94a3b8", margin: 0 }}>
              {records.length}件の解析履歴
            </p>
          </div>
        </div>
      </div>

      {records.length === 0 ? (
        <div
          style={{
            padding: "80px 40px",
            textAlign: "center",
            background: "rgba(255,255,255,0.8)",
            borderRadius: 20,
            border: "1px solid rgba(226,232,240,0.6)",
            backdropFilter: "blur(12px)",
          }}
        >
          <div style={{ fontSize: 48, marginBottom: 16 }}>📊</div>
          <h3
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: "#0f172a",
              marginBottom: 8,
            }}
          >
            まだ履歴がありません
          </h3>
          <p style={{ color: "#64748b", fontSize: 14, lineHeight: 1.6 }}>
            解析を行うと自動的に記録が保存され、
            <br />
            ここにスコアの推移グラフが表示されます
          </p>
        </div>
      ) : (
        <>
          {/* 成長ハイライト */}
          {highlights.length > 0 && (
            <div
              style={{
                marginBottom: 24,
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {highlights.map((h, i) => (
                <div
                  key={i}
                  style={{
                    padding: "14px 20px",
                    background:
                      h.type === "improvement"
                        ? "linear-gradient(135deg, rgba(34,197,94,0.08), rgba(16,185,129,0.04))"
                        : h.type === "stable"
                          ? "linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.04))"
                          : "linear-gradient(135deg, rgba(245,158,11,0.08), rgba(251,191,36,0.04))",
                    borderRadius: 14,
                    border: `1px solid ${
                      h.type === "improvement"
                        ? "rgba(34,197,94,0.2)"
                        : h.type === "stable"
                          ? "rgba(99,102,241,0.2)"
                          : "rgba(245,158,11,0.2)"
                    }`,
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    animation: `slideInRight 0.4s ease ${i * 0.1}s both`,
                  }}
                >
                  <span style={{ fontSize: 20 }}>{h.icon}</span>
                  <span
                    style={{
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#1e293b",
                    }}
                  >
                    {h.message}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* フィルター */}
          <div
            style={{
              display: "flex",
              gap: 8,
              marginBottom: 20,
              flexWrap: "wrap",
            }}
          >
            <button
              onClick={() => setFilterPose("all")}
              style={{
                padding: "6px 16px",
                borderRadius: 20,
                border: "1px solid",
                borderColor:
                  filterPose === "all" ? "#6366f1" : "rgba(226,232,240,0.8)",
                background:
                  filterPose === "all"
                    ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                    : "rgba(255,255,255,0.8)",
                color: filterPose === "all" ? "#fff" : "#64748b",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s ease",
              }}
            >
              すべて
            </button>
            {poseTypes.map((pt) => (
              <button
                key={pt}
                onClick={() => setFilterPose(pt)}
                style={{
                  padding: "6px 16px",
                  borderRadius: 20,
                  border: "1px solid",
                  borderColor:
                    filterPose === pt ? "#6366f1" : "rgba(226,232,240,0.8)",
                  background:
                    filterPose === pt
                      ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                      : "rgba(255,255,255,0.8)",
                  color: filterPose === pt ? "#fff" : "#64748b",
                  fontSize: 12,
                  fontWeight: 600,
                  cursor: "pointer",
                  transition: "all 0.2s ease",
                }}
              >
                {POSE_LABELS[pt] || pt}
              </button>
            ))}
          </div>

          {/* メトリクス選択チップ */}
          <div
            style={{
              display: "flex",
              gap: 6,
              marginBottom: 16,
              flexWrap: "wrap",
            }}
          >
            {availableKeys.map((key) => {
              const active = selectedMetrics.includes(key);
              const color = SCORE_COLORS[key] || "#6366f1";
              return (
                <button
                  key={key}
                  onClick={() => toggleMetric(key)}
                  style={{
                    padding: "4px 12px",
                    borderRadius: 16,
                    border: `1.5px solid ${active ? color : "rgba(226,232,240,0.8)"}`,
                    background: active ? `${color}15` : "rgba(255,255,255,0.6)",
                    color: active ? color : "#94a3b8",
                    fontSize: 11,
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  {active && (
                    <span
                      style={{
                        display: "inline-block",
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: color,
                        marginRight: 5,
                        verticalAlign: "middle",
                      }}
                    />
                  )}
                  {METRIC_LABELS[key] || key}
                </button>
              );
            })}
          </div>

          {/* スコア推移グラフ */}
          <div
            style={{
              background: "rgba(255,255,255,0.85)",
              borderRadius: 20,
              padding: "24px 20px 16px",
              border: "1px solid rgba(226,232,240,0.6)",
              boxShadow:
                "0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06)",
              backdropFilter: "blur(12px)",
              marginBottom: 24,
            }}
          >
            <h3
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "#0f172a",
                marginBottom: 20,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  width: 4,
                  height: 16,
                  borderRadius: 2,
                  background:
                    "linear-gradient(180deg, #6366f1, #8b5cf6)",
                  display: "inline-block",
                }}
              />
              スコア推移
            </h3>
            {chartData.length > 1 ? (
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart
                  data={chartData}
                  margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
                >
                  <defs>
                    {selectedMetrics.map((key) => (
                      <linearGradient
                        key={key}
                        id={`grad_${key}`}
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="5%"
                          stopColor={SCORE_COLORS[key] || "#6366f1"}
                          stopOpacity={0.2}
                        />
                        <stop
                          offset="95%"
                          stopColor={SCORE_COLORS[key] || "#6366f1"}
                          stopOpacity={0}
                        />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(226,232,240,0.5)"
                  />
                  <XAxis
                    dataKey="date"
                    tick={{ fontSize: 11, fill: "#94a3b8" }}
                    axisLine={{ stroke: "rgba(226,232,240,0.5)" }}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 11, fill: "#94a3b8" }}
                    axisLine={{ stroke: "rgba(226,232,240,0.5)" }}
                    tickLine={false}
                    width={35}
                  />
                  <Tooltip
                    contentStyle={{
                      background: "rgba(255,255,255,0.95)",
                      border: "1px solid rgba(226,232,240,0.8)",
                      borderRadius: 12,
                      boxShadow: "0 4px 16px rgba(0,0,0,0.08)",
                      fontSize: 12,
                    }}
                    labelFormatter={(_, payload) => {
                      if (payload && payload.length > 0) {
                        return String(
                          (payload[0].payload as Record<string, string>)
                            .fullDate || ""
                        );
                      }
                      return "";
                    }}
                    formatter={(value: unknown, name: unknown) => [
                      `${Number(value).toFixed(1)}点`,
                      METRIC_LABELS[String(name)] || String(name),
                    ]}
                  />
                  {selectedMetrics.map((key) => (
                    <Area
                      key={key}
                      type="monotone"
                      dataKey={key}
                      stroke={SCORE_COLORS[key] || "#6366f1"}
                      strokeWidth={2.5}
                      fill={`url(#grad_${key})`}
                      dot={{
                        r: 4,
                        fill: SCORE_COLORS[key] || "#6366f1",
                        stroke: "#fff",
                        strokeWidth: 2,
                      }}
                      activeDot={{
                        r: 6,
                        fill: SCORE_COLORS[key] || "#6366f1",
                        stroke: "#fff",
                        strokeWidth: 2,
                      }}
                      connectNulls
                    />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div
                style={{
                  padding: "40px 0",
                  textAlign: "center",
                  color: "#94a3b8",
                  fontSize: 13,
                }}
              >
                グラフを表示するには2件以上のデータが必要です
              </div>
            )}
          </div>

          {/* 履歴リスト */}
          <div
            style={{
              background: "rgba(255,255,255,0.85)",
              borderRadius: 20,
              padding: 24,
              border: "1px solid rgba(226,232,240,0.6)",
              boxShadow:
                "0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06)",
              backdropFilter: "blur(12px)",
            }}
          >
            <h3
              style={{
                fontSize: 14,
                fontWeight: 700,
                color: "#0f172a",
                marginBottom: 16,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span
                style={{
                  width: 4,
                  height: 16,
                  borderRadius: 2,
                  background: "linear-gradient(180deg, #10b981, #06b6d4)",
                  display: "inline-block",
                }}
              />
              解析履歴
            </h3>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              {filtered
                .slice()
                .reverse()
                .map((rec, i) => (
                  <HistoryRow key={rec.id} record={rec} index={i} />
                ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function scoreColor(score: number): string {
  if (score >= 85) return "#22c55e";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

function HistoryRow({
  record,
  index,
}: {
  record: HistoryRecord;
  index: number;
}) {
  const [hover, setHover] = useState(false);
  const color = scoreColor(record.overall_score);
  const poseName =
    POSE_LABELS[record.pose_type] || record.pose_type;

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "12px 16px",
        borderRadius: 12,
        background: hover
          ? "rgba(248,250,252,1)"
          : "rgba(248,250,252,0.5)",
        border: "1px solid rgba(226,232,240,0.4)",
        transition: "all 0.2s ease",
        transform: hover ? "translateX(4px)" : "none",
        animation: `slideInRight 0.3s ease ${index * 0.05}s both`,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {/* スコアバッジ */}
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: 12,
            background: `${color}15`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontWeight: 800,
            fontSize: 16,
            color,
          }}
        >
          {record.overall_score.toFixed(0)}
        </div>
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 2,
            }}
          >
            <span
              style={{
                fontSize: 13,
                fontWeight: 700,
                color: "#0f172a",
              }}
            >
              {poseName}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                color: "#6366f1",
                background: "rgba(99,102,241,0.08)",
                padding: "1px 8px",
                borderRadius: 6,
              }}
            >
              {record.pose_type}
            </span>
          </div>
          <span style={{ fontSize: 11, color: "#94a3b8" }}>
            {formatDate(record.created_at)}
          </span>
        </div>
      </div>

      {/* ミニスコアバー */}
      <div style={{ display: "flex", gap: 4 }}>
        {Object.entries(record.scores)
          .slice(0, 4)
          .map(([key, val]) => (
            <div
              key={key}
              style={{
                width: 32,
                height: 4,
                borderRadius: 2,
                background: `${scoreColor(val)}40`,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${val}%`,
                  height: "100%",
                  background: scoreColor(val),
                  borderRadius: 2,
                }}
              />
            </div>
          ))}
      </div>
    </div>
  );
}
