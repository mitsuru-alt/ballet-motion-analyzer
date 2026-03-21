import { useState } from "react";
import type { AdviceItem } from "../hooks/useAnalysis";

const LEVEL_STYLES: Record<
  string,
  { bg: string; border: string; icon: string; iconColor: string; iconBg: string }
> = {
  excellent: {
    bg: "rgba(240,253,244,0.8)",
    border: "rgba(187,247,208,0.6)",
    icon: "\u2713",
    iconColor: "#16a34a",
    iconBg: "rgba(22,163,74,0.1)",
  },
  good: {
    bg: "rgba(254,252,232,0.8)",
    border: "rgba(253,230,138,0.6)",
    icon: "\u25B2",
    iconColor: "#ca8a04",
    iconBg: "rgba(202,138,4,0.1)",
  },
  needs_work: {
    bg: "rgba(254,242,242,0.8)",
    border: "rgba(254,202,202,0.6)",
    icon: "\u25CF",
    iconColor: "#dc2626",
    iconBg: "rgba(220,38,38,0.1)",
  },
};

/** ブロック見出しとアイコン・色の対応 */
const BLOCK_HEADER_CONFIG: Record<
  string,
  { icon: string; color: string; bg: string }
> = {
  "すばらしい！": { icon: "🌟", color: "#16a34a", bg: "rgba(22,163,74,0.08)" },
  "現状": { icon: "📐", color: "#6366f1", bg: "rgba(99,102,241,0.08)" },
  "意識するイメージ": { icon: "💡", color: "#d97706", bg: "rgba(217,119,6,0.08)" },
  "さらに磨くなら": { icon: "💎", color: "#8b5cf6", bg: "rgba(139,92,246,0.08)" },
  "おすすめの練習": { icon: "🩰", color: "#0891b2", bg: "rgba(8,145,178,0.08)" },
};

/** 「」内のテキストをハイライトする */
function renderWithQuoteHighlight(text: string, accentColor: string) {
  const parts = text.split(/(「[^」]*」)/g);
  return parts.map((part, i) => {
    if (part.startsWith("「") && part.endsWith("」")) {
      return (
        <span
          key={i}
          style={{
            color: accentColor,
            fontWeight: 600,
            fontStyle: "italic",
          }}
        >
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

/** メッセージを3ブロック構造としてパース */
function parseBlocks(message: string) {
  const rawBlocks = message.split("\n\n");
  const blocks: { header: string; body: string }[] = [];

  for (const raw of rawBlocks) {
    const match = raw.match(/^【([^】]+)】\n?([\s\S]*)$/);
    if (match) {
      blocks.push({ header: match[1], body: match[2].trim() });
    } else if (raw.trim()) {
      // フォールバック: ヘッダーなしブロック
      blocks.push({ header: "", body: raw.trim() });
    }
  }

  return blocks;
}

interface Props {
  advice: AdviceItem[];
}

export default function AdviceCard({ advice }: Props) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  if (advice.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <div
          style={{
            width: 4,
            height: 20,
            borderRadius: 2,
            background: "linear-gradient(180deg, #6366f1, #8b5cf6)",
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
          改善アドバイス
        </h3>
        <span
          style={{
            fontSize: 11,
            color: "#6366f1",
            fontWeight: 600,
            background: "rgba(99,102,241,0.08)",
            padding: "2px 8px",
            borderRadius: 6,
          }}
        >
          {advice.length}項目
        </span>
      </div>

      {advice.map((item, i) => {
        const style = LEVEL_STYLES[item.level] ?? LEVEL_STYLES.needs_work;
        const isHovered = hoveredIdx === i;
        const blocks = parseBlocks(item.message);
        const hasBlocks = blocks.length > 0 && blocks[0].header !== "";

        return (
          <div
            key={i}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            style={{
              padding: "16px 18px",
              background: style.bg,
              border: `1px solid ${style.border}`,
              borderLeft: `4px solid ${style.iconColor}`,
              borderRadius: 14,
              backdropFilter: "blur(4px)",
              transition: "all 0.2s cubic-bezier(0.4,0,0.2,1)",
              animation: `slideInRight 0.4s ease ${0.1 + i * 0.08}s both`,
              transform: isHovered ? "translateX(4px)" : "none",
              boxShadow: isHovered
                ? "0 4px 16px rgba(0,0,0,0.06)"
                : "none",
            }}
          >
            {/* Header */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: hasBlocks ? 12 : 8,
              }}
            >
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 26,
                  height: 26,
                  borderRadius: 8,
                  background: style.iconBg,
                  color: style.iconColor,
                  fontSize: 13,
                  fontWeight: 700,
                  border: `1px solid ${style.iconColor}30`,
                }}
              >
                {style.icon}
              </span>
              <span
                style={{ fontWeight: 600, fontSize: 14, color: "#374151" }}
              >
                {item.metric_name}
              </span>
              <span
                style={{
                  marginLeft: "auto",
                  fontSize: 13,
                  fontWeight: 600,
                  color: style.iconColor,
                }}
              >
                {item.score.toFixed(0)} / 100
              </span>
            </div>

            {/* Advice content */}
            {hasBlocks ? (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {blocks.map((block, bi) => {
                  const config = BLOCK_HEADER_CONFIG[block.header];
                  if (!config) {
                    // フォールバック
                    return (
                      <p
                        key={bi}
                        style={{
                          margin: 0,
                          fontSize: 13,
                          lineHeight: 1.7,
                          color: "#4b5563",
                        }}
                      >
                        {block.body}
                      </p>
                    );
                  }
                  return (
                    <div
                      key={bi}
                      style={{
                        background: config.bg,
                        borderRadius: 10,
                        padding: "10px 12px",
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
                        <span style={{ fontSize: 14 }}>{config.icon}</span>
                        <span
                          style={{
                            fontSize: 12,
                            fontWeight: 700,
                            color: config.color,
                            letterSpacing: "0.02em",
                          }}
                        >
                          {block.header}
                        </span>
                      </div>
                      <p
                        style={{
                          margin: 0,
                          fontSize: 13,
                          lineHeight: 1.8,
                          color: "#374151",
                        }}
                      >
                        {renderWithQuoteHighlight(block.body, config.color)}
                      </p>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p
                style={{
                  margin: 0,
                  fontSize: 13,
                  lineHeight: 1.7,
                  color: "#4b5563",
                }}
              >
                {item.message}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}
