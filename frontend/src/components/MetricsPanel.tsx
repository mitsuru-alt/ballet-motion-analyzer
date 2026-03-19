import { useState, useEffect } from "react";
import type { AnalysisResponse } from "../hooks/useAnalysis";

const METRIC_LABELS: Record<string, string> = {
  standing_knee: "軸脚の膝",
  releve_height: "ルルヴェの高さ",
  vertical_axis: "体幹の垂直性",
  arm_position: "腕のポジション",
  pelvic_level: "骨盤の水平性",
  working_leg: "パッセ脚の位置",
};

const POSE_LABELS: Record<string, string> = {
  pirouette: "ピルエット",
  unknown: "未検出",
};

function scoreColor(score: number): string {
  if (score >= 85) return "#22c55e";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

function scoreBg(score: number): string {
  if (score >= 85) return "rgba(240,253,244,0.8)";
  if (score >= 60) return "rgba(254,252,232,0.8)";
  return "rgba(254,242,242,0.8)";
}

function scoreEmoji(score: number): string {
  if (score >= 90) return "Excellent";
  if (score >= 75) return "Good";
  if (score >= 60) return "Fair";
  return "Needs Work";
}

interface Props {
  data: AnalysisResponse;
}

export default function MetricsPanel({ data }: Props) {
  const circumference = 2 * Math.PI * 54;
  const offset = circumference * (1 - data.overall_score / 100);
  const entries = Object.entries(data.scores);

  // Score count-up animation
  const [displayScore, setDisplayScore] = useState(0);
  useEffect(() => {
    const target = data.overall_score;
    const duration = 1000;
    const startTime = Date.now();
    let raf: number;

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayScore(Math.round(eased * target));
      if (progress < 1) raf = requestAnimationFrame(animate);
    };
    raf = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(raf);
  }, [data.overall_score]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Pose type badge */}
      <div style={{ textAlign: "center" }}>
        <span
          style={{
            display: "inline-block",
            padding: "6px 20px",
            background:
              "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 50%, #ddd6fe 100%)",
            color: "#4338ca",
            borderRadius: 24,
            fontSize: 13,
            fontWeight: 700,
            letterSpacing: "0.5px",
            boxShadow: "0 2px 8px rgba(99,102,241,0.12)",
            border: "1px solid rgba(165,180,252,0.3)",
            animation: "fadeInScale 0.4s ease",
          }}
        >
          {POSE_LABELS[data.pose_type] ?? data.pose_type}
        </span>
      </div>

      {/* Score ring chart */}
      <div style={{ textAlign: "center" }}>
        <svg width={150} height={150} viewBox="0 0 150 150">
          <defs>
            <filter id="ringGlow">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>
          {/* Background ring */}
          <circle
            cx={75}
            cy={75}
            r={54}
            fill="none"
            stroke="rgba(243,244,246,0.8)"
            strokeWidth={10}
          />
          {/* Score ring */}
          <circle
            cx={75}
            cy={75}
            r={54}
            fill="none"
            stroke={scoreColor(data.overall_score)}
            strokeWidth={10}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            transform="rotate(-90 75 75)"
            filter="url(#ringGlow)"
            style={{
              transition:
                "stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)",
              animation: "ringDraw 1.2s cubic-bezier(0.4,0,0.2,1)",
            }}
          />
          {/* Score number */}
          <text
            x={75}
            y={65}
            textAnchor="middle"
            fontSize={36}
            fontWeight="800"
            fill="#0f172a"
          >
            {displayScore}
          </text>
          <text
            x={75}
            y={88}
            textAnchor="middle"
            fontSize={11}
            fill="#94a3b8"
            fontWeight="600"
            letterSpacing="0.05em"
          >
            {scoreEmoji(data.overall_score)}
          </text>
        </svg>
      </div>

      {/* Radar chart */}
      {entries.length >= 3 && <RadarChart entries={entries} />}

      {/* Individual metric bars */}
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {entries.map(([key, score], index) => (
          <div
            key={key}
            style={{
              animation: `slideInRight 0.4s ease ${0.1 + index * 0.06}s both`,
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 13,
                marginBottom: 5,
              }}
            >
              <span style={{ color: "#374151", fontWeight: 500 }}>
                {METRIC_LABELS[key] ?? key}
              </span>
              <span
                style={{
                  fontWeight: 700,
                  color: scoreColor(score),
                  fontSize: 14,
                }}
              >
                {score.toFixed(0)}
              </span>
            </div>
            <div
              style={{
                height: 8,
                background: "rgba(243,244,246,0.8)",
                borderRadius: 6,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${Math.min(100, score)}%`,
                  height: "100%",
                  background: `linear-gradient(90deg, ${scoreColor(score)}99, ${scoreColor(score)})`,
                  borderRadius: 6,
                  transition:
                    "width 0.6s cubic-bezier(0.4,0,0.2,1)",
                  animation: `barGrow 0.8s cubic-bezier(0.4,0,0.2,1) ${0.3 + index * 0.08}s both`,
                  boxShadow: `0 0 8px ${scoreColor(score)}33`,
                }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Stability (video only) */}
      {data.stability && (
        <div
          style={{
            padding: 14,
            background: scoreBg(data.stability.stability_score),
            borderRadius: 12,
            fontSize: 13,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            border: `1px solid ${scoreColor(data.stability.stability_score)}20`,
            backdropFilter: "blur(4px)",
          }}
        >
          <span style={{ fontWeight: 600, color: "#374151" }}>重心安定性</span>
          <span
            style={{
              color: scoreColor(data.stability.stability_score),
              fontWeight: 700,
              fontSize: 16,
            }}
          >
            {data.stability.stability_score.toFixed(0)}
            <span
              style={{ fontSize: 11, color: "#9ca3af", fontWeight: 400 }}
            >
              {" "}
              / 100
            </span>
          </span>
        </div>
      )}
    </div>
  );
}

/* ================================================================
 *  Radar Chart
 * ================================================================ */

function RadarChart({ entries }: { entries: [string, number][] }) {
  const margin = 50;
  const maxR = 65;
  const size = (maxR + margin) * 2;
  const cx = size / 2;
  const cy = size / 2;
  const n = entries.length;

  const angleStep = (2 * Math.PI) / n;
  const startAngle = -Math.PI / 2;

  const gridLevels = [25, 50, 75, 100];
  const gridPaths = gridLevels.map((level) => {
    const r = (level / 100) * maxR;
    const points = Array.from({ length: n }, (_, i) => {
      const angle = startAngle + i * angleStep;
      return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
    });
    return points.join(" ");
  });

  const dataPoints = entries.map(([, score], i) => {
    const r = (Math.min(100, score) / 100) * maxR;
    const angle = startAngle + i * angleStep;
    return `${cx + r * Math.cos(angle)},${cy + r * Math.sin(angle)}`;
  });

  const axes = entries.map(([key], i) => {
    const angle = startAngle + i * angleStep;
    const labelR = maxR + 28;
    const cosA = Math.cos(angle);
    let anchor: "start" | "middle" | "end" = "middle";
    if (cosA > 0.3) anchor = "start";
    else if (cosA < -0.3) anchor = "end";

    return {
      x1: cx,
      y1: cy,
      x2: cx + maxR * Math.cos(angle),
      y2: cy + maxR * Math.sin(angle),
      lx: cx + labelR * Math.cos(angle),
      ly: cy + labelR * Math.sin(angle),
      label: METRIC_LABELS[key] ?? key,
      anchor,
    };
  });

  return (
    <div style={{ textAlign: "center" }}>
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ overflow: "visible" }}
      >
        <defs>
          <linearGradient id="radarFill" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="rgba(99,102,241,0.25)" />
            <stop offset="100%" stopColor="rgba(139,92,246,0.15)" />
          </linearGradient>
        </defs>
        {/* Grid */}
        {gridPaths.map((points, i) => (
          <polygon
            key={i}
            points={points}
            fill="none"
            stroke="rgba(229,231,235,0.6)"
            strokeWidth={0.8}
          />
        ))}
        {/* Axes */}
        {axes.map((a, i) => (
          <line
            key={i}
            x1={a.x1}
            y1={a.y1}
            x2={a.x2}
            y2={a.y2}
            stroke="rgba(209,213,219,0.6)"
            strokeWidth={0.8}
          />
        ))}
        {/* Data polygon */}
        <polygon
          points={dataPoints.join(" ")}
          fill="url(#radarFill)"
          stroke="#6366f1"
          strokeWidth={2.5}
        />
        {/* Data vertices */}
        {entries.map(([, score], i) => {
          const r = (Math.min(100, score) / 100) * maxR;
          const angle = startAngle + i * angleStep;
          return (
            <circle
              key={i}
              cx={cx + r * Math.cos(angle)}
              cy={cy + r * Math.sin(angle)}
              r={4}
              fill={scoreColor(score)}
              stroke="#fff"
              strokeWidth={2}
              style={{
                filter: "drop-shadow(0 0 3px rgba(99,102,241,0.3))",
              }}
            />
          );
        })}
        {/* Labels */}
        {axes.map((a, i) => (
          <text
            key={i}
            x={a.lx}
            y={a.ly}
            textAnchor={a.anchor}
            dominantBaseline="central"
            fontSize={10}
            fill="#4b5563"
            fontWeight="600"
          >
            {a.label}
          </text>
        ))}
      </svg>
    </div>
  );
}
