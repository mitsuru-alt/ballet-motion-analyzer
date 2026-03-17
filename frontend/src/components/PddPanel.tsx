import { useState } from "react";
import type { PairData } from "../hooks/useAnalysis";

interface Props {
  data: PairData;
}

function scoreColor(score: number): string {
  if (score >= 85) return "#22c55e";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

/** 円形ゲージ SVG */
function CircleGauge({
  value,
  max,
  label,
  unit,
  color,
  size = 100,
}: {
  value: number;
  max: number;
  label: string;
  unit: string;
  color: string;
  size?: number;
}) {
  const strokeWidth = 7;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const progress = Math.min(value / max, 1);
  const offset = circumference * (1 - progress);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 6,
      }}
    >
      <svg width={size} height={size} style={{ transform: "rotate(-90deg)" }}>
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(226,232,240,0.5)"
          strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{
            transition: "stroke-dashoffset 1s cubic-bezier(0.4,0,0.2,1)",
            filter: `drop-shadow(0 0 4px ${color}40)`,
          }}
        />
      </svg>
      <div
        style={{
          position: "relative",
          marginTop: -size - 6,
          height: size,
          width: size,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          pointerEvents: "none",
        }}
      >
        <span
          style={{
            fontSize: 22,
            fontWeight: 800,
            color: "#0f172a",
            lineHeight: 1,
          }}
        >
          {value.toFixed(0)}
        </span>
        <span
          style={{ fontSize: 10, color: "#94a3b8", fontWeight: 500 }}
        >
          {unit}
        </span>
      </div>
      <span
        style={{
          fontSize: 11,
          fontWeight: 600,
          color: "#64748b",
          textAlign: "center",
          marginTop: 2,
        }}
      >
        {label}
      </span>
    </div>
  );
}

/** サポート距離のバー */
function DistanceBar({
  value,
  score,
}: {
  value: number;
  score: number;
}) {
  // 理想範囲: 0.15-0.30
  const max = 0.50;
  const idealLeft = 0.15 / max;
  const idealRight = 0.30 / max;
  const pos = Math.min(value / max, 1);
  const color = scoreColor(score);

  return (
    <div style={{ padding: "0 4px" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 10,
          color: "#94a3b8",
          marginBottom: 4,
        }}
      >
        <span>近い</span>
        <span>理想 (0.15-0.30)</span>
        <span>遠い</span>
      </div>
      <div
        style={{
          position: "relative",
          height: 10,
          background: "rgba(226,232,240,0.5)",
          borderRadius: 5,
          overflow: "visible",
        }}
      >
        {/* 理想範囲 */}
        <div
          style={{
            position: "absolute",
            left: `${idealLeft * 100}%`,
            width: `${(idealRight - idealLeft) * 100}%`,
            height: "100%",
            background: "rgba(34,197,94,0.2)",
            borderRadius: 5,
          }}
        />
        {/* 現在位置マーカー */}
        <div
          style={{
            position: "absolute",
            left: `${pos * 100}%`,
            top: -3,
            width: 16,
            height: 16,
            borderRadius: "50%",
            background: color,
            border: "2px solid white",
            boxShadow: `0 0 8px ${color}66`,
            transform: "translateX(-50%)",
            transition: "left 0.5s ease",
          }}
        />
      </div>
      <div
        style={{
          textAlign: "center",
          fontSize: 12,
          fontWeight: 700,
          color,
          marginTop: 6,
        }}
      >
        {value.toFixed(3)}
      </div>
    </div>
  );
}

export default function PddPanel({ data }: Props) {
  const [, setIsHovered] = useState(false);
  const pm = data.pair_metrics;
  const ps = data.pair_scores;

  const comScore = ps?.shared_com ?? 0;
  const trunkScore = ps?.trunk_angle ?? 0;
  const distScore = ps?.support_distance ?? 0;

  return (
    <div
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {/* Section header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            width: 4,
            height: 20,
            borderRadius: 2,
            background: "linear-gradient(180deg, #10b981, #3b82f6)",
          }}
        />
        <h3
          style={{
            fontSize: 16,
            fontWeight: 700,
            color: "#0f172a",
            margin: 0,
            letterSpacing: "-0.01em",
          }}
        >
          パ・ド・ドゥ解析
        </h3>
        <span
          style={{
            fontSize: 11,
            color: "#10b981",
            fontWeight: 600,
            background: "rgba(16,185,129,0.08)",
            padding: "2px 8px",
            borderRadius: 6,
          }}
        >
          Pair
        </span>
      </div>

      {/* ダンサーカラー凡例 */}
      <div
        style={{
          display: "flex",
          gap: 16,
          justifyContent: "center",
          fontSize: 12,
          color: "#64748b",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#10b981",
            }}
          />
          <span>Dancer A</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <div
            style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: "#3b82f6",
            }}
          />
          <span>Dancer B</span>
        </div>
      </div>

      {/* 円形ゲージ */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-around",
          alignItems: "flex-start",
          padding: "4px 0",
        }}
      >
        <CircleGauge
          value={comScore}
          max={100}
          label="重心安定性"
          unit="点"
          color={scoreColor(comScore)}
          size={96}
        />
        <CircleGauge
          value={trunkScore}
          max={100}
          label="体幹垂直性"
          unit="点"
          color={scoreColor(trunkScore)}
          size={96}
        />
        <CircleGauge
          value={distScore}
          max={100}
          label="距離感"
          unit="点"
          color={scoreColor(distScore)}
          size={96}
        />
      </div>

      {/* サポート距離バー */}
      <div
        style={{
          padding: "10px 12px",
          background: "rgba(248,250,252,0.8)",
          borderRadius: 10,
          border: "1px solid rgba(226,232,240,0.5)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 6,
            marginBottom: 8,
          }}
        >
          <span style={{ fontSize: 12 }}>📏</span>
          <span
            style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}
          >
            サポート距離
          </span>
        </div>
        <DistanceBar value={pm.support_distance} score={distScore} />
      </div>

      {/* 詳細数値 */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
        }}
      >
        {[
          {
            label: "体幹角度",
            value: `${pm.trunk_verticality.toFixed(1)}°`,
            icon: "📐",
          },
          {
            label: "重心変位",
            value: pm.shared_com_displacement.toFixed(4),
            icon: "⚖️",
          },
          {
            label: "基底内",
            value: pm.within_base ? "安定 ✓" : "外れ ✗",
            icon: pm.within_base ? "🟢" : "🔴",
          },
          {
            label: "サポート対象",
            value: pm.supported_person_id === 0 ? "Dancer A" : "Dancer B",
            icon: "🩰",
          },
        ].map((stat, i) => (
          <div
            key={i}
            style={{
              padding: "10px 12px",
              background: "rgba(248,250,252,0.8)",
              borderRadius: 10,
              border: "1px solid rgba(226,232,240,0.5)",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                marginBottom: 4,
              }}
            >
              <span style={{ fontSize: 12 }}>{stat.icon}</span>
              <span
                style={{ fontSize: 10, color: "#94a3b8", fontWeight: 600 }}
              >
                {stat.label}
              </span>
            </div>
            <span
              style={{ fontSize: 13, fontWeight: 700, color: "#1e293b" }}
            >
              {stat.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
