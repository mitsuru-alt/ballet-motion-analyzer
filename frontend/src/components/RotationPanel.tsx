import { useState } from "react";

interface RotationData {
  rotation_count: number;
  avg_seconds_per_turn: number;
  rpm: number;
  rotation_start_ms: number;
  rotation_end_ms: number;
  rotation_duration_ms: number;
  peak_speed_rpm: number;
}

interface Props {
  data: RotationData;
}

/** 回転数に応じた励ましテキスト */
function getMotivationText(count: number): {
  title: string;
  message: string;
  color: string;
} {
  if (count >= 3) {
    return {
      title: "トリプル達成！",
      message: "素晴らしい回転です！安定感のあるトリプルは大きな武器になります。",
      color: "#16a34a",
    };
  }
  if (count >= 2.5) {
    return {
      title: "あと0.5回転でトリプル！",
      message:
        "もう手が届いています！スポットを意識して、最後まで回りきりましょう。",
      color: "#8b5cf6",
    };
  }
  if (count >= 2) {
    return {
      title: "ダブル達成！",
      message:
        "安定したダブルは素晴らしい成果です。次はトリプルを目指しましょう！",
      color: "#6366f1",
    };
  }
  if (count >= 1.5) {
    return {
      title: "あと0.5回転でダブル！",
      message:
        "もうすぐダブルです！プリエを深くして、しっかり床を押して回りましょう。",
      color: "#0891b2",
    };
  }
  if (count >= 1) {
    return {
      title: "シングル達成！",
      message:
        "しっかり1回転できています！まずは安定したシングルを磨きましょう。",
      color: "#d97706",
    };
  }
  return {
    title: "回転の始まり",
    message:
      "回転の動きが見えています。軸を安定させることから始めましょう！",
    color: "#64748b",
  };
}

/** 次の目標を自動判定 */
function getNextGoal(count: number): { label: string; target: number } {
  if (count >= 3) return { label: "トリプル+", target: 4 };
  if (count >= 2) return { label: "トリプル", target: 3 };
  if (count >= 1) return { label: "ダブル", target: 2 };
  return { label: "シングル", target: 1 };
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
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(226,232,240,0.5)"
          strokeWidth={strokeWidth}
        />
        {/* Progress arc */}
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
      {/* Center text overlay */}
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
          {typeof value === "number" && value % 1 === 0
            ? value.toFixed(0)
            : value.toFixed(1)}
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

export default function RotationPanel({ data }: Props) {
  const [, setIsHovered] = useState(false);
  const motivation = getMotivationText(data.rotation_count);
  const nextGoal = getNextGoal(data.rotation_count);

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
            background: "linear-gradient(180deg, #ec4899, #f43f5e)",
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
          回転解析
        </h3>
        <span
          style={{
            fontSize: 11,
            color: "#ec4899",
            fontWeight: 600,
            background: "rgba(236,72,153,0.08)",
            padding: "2px 8px",
            borderRadius: 6,
          }}
        >
          Motion
        </span>
      </div>

      {/* Circular gauges row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-around",
          alignItems: "flex-start",
          padding: "8px 0",
        }}
      >
        <CircleGauge
          value={data.rotation_count}
          max={nextGoal.target}
          label={`目標: ${nextGoal.label}`}
          unit="回転"
          color="#ec4899"
          size={100}
        />
        <CircleGauge
          value={data.rpm}
          max={180}
          label="回転速度"
          unit="RPM"
          color="#8b5cf6"
          size={100}
        />
      </div>

      {/* Detail stats grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
        }}
      >
        {[
          {
            label: "平均速度",
            value:
              data.avg_seconds_per_turn > 0
                ? `${data.avg_seconds_per_turn}秒/回転`
                : "—",
            icon: "⏱",
          },
          {
            label: "ピーク速度",
            value:
              data.peak_speed_rpm > 0
                ? `${data.peak_speed_rpm.toFixed(0)} RPM`
                : "—",
            icon: "⚡",
          },
          {
            label: "回転区間",
            value:
              data.rotation_duration_ms > 0
                ? `${(data.rotation_duration_ms / 1000).toFixed(1)}秒`
                : "—",
            icon: "📐",
          },
          {
            label: "回転数",
            value:
              data.rotation_count > 0
                ? `${data.rotation_count}回転`
                : "検出なし",
            icon: "🔄",
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

      {/* Motivation message */}
      <div
        style={{
          padding: "12px 14px",
          background: `linear-gradient(135deg, ${motivation.color}08, ${motivation.color}04)`,
          borderRadius: 12,
          borderLeft: `3px solid ${motivation.color}`,
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
          <span style={{ fontSize: 14 }}>
            {data.rotation_count >= 3
              ? "🎉"
              : data.rotation_count >= 2
                ? "✨"
                : data.rotation_count >= 1
                  ? "💪"
                  : "🌱"}
          </span>
          <span
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: motivation.color,
            }}
          >
            {motivation.title}
          </span>
        </div>
        <p
          style={{
            margin: 0,
            fontSize: 12,
            lineHeight: 1.7,
            color: "#4b5563",
          }}
        >
          {motivation.message}
        </p>
      </div>
    </div>
  );
}
