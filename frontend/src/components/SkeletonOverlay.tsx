import { useEffect, useRef } from "react";
import type { FrameLandmarks, PairData } from "../hooks/useAnalysis";

/**
 * SkeletonOverlay - Canvas上に骨格・角度アーク・プロ基準をオーバーレイ描画
 *
 * 描画レイヤー (下から):
 *   1. 元画像 (薄暗く)
 *   2. 骨格の接続線 (白、グロー効果)
 *   3. 関節点 (色分けした円)
 *   4. 扇形の角度アーク (評価対象の関節に描画)
 *   5. プロ基準のゴースト線 (理想角度を半透明で重ねる)
 *   6. スコアラベル (角度値 + スコア)
 */

const CONNECTIONS: [number, number][] = [
  [11, 12], [11, 23], [12, 24], [23, 24],
  [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [27, 31], [28, 32],
  [11, 13], [12, 14], [13, 15], [14, 16],
];

// 骨格接続線の色: 体幹=シアン, 脚=マゼンタ, 腕=ライム
function connectionColor(a: number, b: number): string {
  if ([11, 12, 23, 24].includes(a) && [11, 12, 23, 24].includes(b)) return "#06b6d4";
  if ([23, 24, 25, 26, 27, 28, 29, 30, 31, 32].includes(a)) return "#ec4899";
  return "#a3e635";
}

function scoreColor(score: number): string {
  if (score >= 85) return "#22c55e";
  if (score >= 60) return "#f59e0b";
  return "#ef4444";
}

// パ・ド・ドゥ: ダンサー別カラー
const PAIR_COLORS = {
  0: { main: "#10b981", glow: "#34d399", label: "Dancer A" },  // エメラルドグリーン
  1: { main: "#3b82f6", glow: "#60a5fa", label: "Dancer B" },  // サファイアブルー
} as const;

interface Props {
  imageSrc: string;
  frameData: FrameLandmarks;
  scores: Record<string, number>;
  metrics: Record<string, number | string>;
  poseType: string;
  width?: number;
  pairData?: PairData;
}

export default function SkeletonOverlay({
  imageSrc,
  frameData,
  scores,
  metrics,
  poseType,
  width = 660,
  pairData,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;

    const draw = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const scale = width / img.naturalWidth;
      const h = img.naturalHeight * scale;
      canvas.width = width;
      canvas.height = h;

      // Layer 1: 元画像 (やや暗く)
      ctx.drawImage(img, 0, 0, width, h);
      ctx.fillStyle = "rgba(0,0,0,0.25)";
      ctx.fillRect(0, 0, width, h);

      // === パ・ド・ドゥモード: 2人の骨格を色分けして描画 ===
      if (poseType === "pas_de_deux" && pairData && pairData.persons.length >= 2) {
        drawPairSkeletons(ctx, pairData, width, h);
        drawSharedCoM(ctx, pairData, scores, width, h);
        drawSupportDistanceLine(ctx, pairData, scores, width, h);
        return; // ペアモードでは個別ポーズオーバーレイは不要
      }

      // === 単一人物モード（既存ロジック）===
      const lm = frameData.landmarks;
      const px = (i: number) => lm[i].x * width;
      const py = (i: number) => lm[i].y * h;
      const vis = (i: number) => (lm[i]?.visibility ?? 0) > 0.5;

      // Layer 2: 骨格接続線 (グロー付き)
      for (const [a, b] of CONNECTIONS) {
        if (!vis(a) || !vis(b)) continue;
        // グロー
        ctx.save();
        ctx.shadowColor = connectionColor(a, b);
        ctx.shadowBlur = 8;
        ctx.strokeStyle = connectionColor(a, b);
        ctx.lineWidth = 3;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.moveTo(px(a), py(a));
        ctx.lineTo(px(b), py(b));
        ctx.stroke();
        ctx.restore();
        // 実線
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.moveTo(px(a), py(a));
        ctx.lineTo(px(b), py(b));
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;

      // Layer 3: 関節点
      const jointMap: Record<number, string> = {
        11: "#3b82f6", 12: "#3b82f6",
        23: "#22c55e", 24: "#22c55e",
        25: "#f97316", 26: "#f97316",
        27: "#ef4444", 28: "#ef4444",
      };
      for (const idx of [11, 12, 23, 24, 25, 26, 27, 28, 13, 14, 15, 16, 29, 30, 31, 32]) {
        if (!vis(idx)) continue;
        const x = px(idx);
        const y = py(idx);
        const color = jointMap[idx] ?? "#a855f7";
        const r = [11, 12, 23, 24, 25, 26, 27, 28].includes(idx) ? 7 : 4;

        // グロー
        ctx.save();
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // 白縁
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Layer 4 & 5: 扇形角度アーク + プロ基準ゴーストライン + ラベル
      if (poseType === "arabesque") {
        drawArabesqueOverlays(ctx, lm, scores, metrics, width, h);
      } else if (poseType === "passe") {
        drawPasseOverlays(ctx, lm, scores, metrics, width, h);
      } else if (poseType === "pirouette") {
        drawPirouetteOverlays(ctx, lm, scores, metrics, width, h);
      }
    };

    if (img.complete) {
      draw();
    } else {
      img.onload = draw;
    }
  }, [imageSrc, frameData, scores, metrics, poseType, width, pairData]);

  return (
    <div
      style={{
        position: "relative",
        display: "inline-block",
        borderRadius: 16,
        padding: 2,
        background:
          "linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.1), rgba(236,72,153,0.1))",
      }}
    >
      <img
        ref={imgRef}
        src={imageSrc}
        alt="source"
        style={{ display: "none" }}
        crossOrigin="anonymous"
      />
      <canvas
        ref={canvasRef}
        style={{
          borderRadius: 14,
          maxWidth: "100%",
          height: "auto",
          display: "block",
          boxShadow:
            "0 8px 32px rgba(0,0,0,0.2), 0 2px 8px rgba(0,0,0,0.1)",
        }}
      />
    </div>
  );
}

/* ================================================================
 *  角度アーク描画ユーティリティ
 * ================================================================ */

type LM = FrameLandmarks["landmarks"];

function drawArc(
  ctx: CanvasRenderingContext2D,
  cx: number, cy: number,
  radius: number,
  startAngle: number,
  endAngle: number,
  color: string,
  label: string,
  score: number,
) {
  // 扇形の塗りつぶし
  ctx.save();
  ctx.globalAlpha = 0.25;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.arc(cx, cy, radius, startAngle, endAngle);
  ctx.closePath();
  ctx.fill();
  ctx.restore();

  // 扇形の枠線
  ctx.save();
  ctx.globalAlpha = 0.9;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, startAngle, endAngle);
  ctx.stroke();
  ctx.restore();

  // ラベル (アークの中間角に配置)
  const midAngle = (startAngle + endAngle) / 2;
  const lx = cx + (radius + 18) * Math.cos(midAngle);
  const ly = cy + (radius + 18) * Math.sin(midAngle);
  drawLabel(ctx, lx, ly, label, score);
}

function drawLabel(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  text: string,
  score: number,
) {
  const color = scoreColor(score);
  ctx.save();
  ctx.font = "bold 13px 'Helvetica Neue', sans-serif";
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  const m = ctx.measureText(text);
  const pad = 6;
  const bw = m.width + pad * 2;
  const bh = 22;

  // 背景ピル
  ctx.fillStyle = "rgba(0,0,0,0.75)";
  ctx.beginPath();
  ctx.roundRect(x - bw / 2, y - bh / 2, bw, bh, bh / 2);
  ctx.fill();

  // テキスト
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawProGhostLine(
  ctx: CanvasRenderingContext2D,
  originX: number, originY: number,
  targetAngle: number, // ラジアン
  length: number,
  label: string,
) {
  const ex = originX + length * Math.cos(targetAngle);
  const ey = originY + length * Math.sin(targetAngle);

  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = "rgba(34, 211, 238, 0.6)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(originX, originY);
  ctx.lineTo(ex, ey);
  ctx.stroke();
  ctx.setLineDash([]);

  // ラベル
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(34, 211, 238, 0.8)";
  ctx.textAlign = "center";
  ctx.fillText(label, ex, ey - 8);
  ctx.restore();
}

/* ================================================================
 *  アラベスク用オーバーレイ
 * ================================================================ */

function drawArabesqueOverlays(
  ctx: CanvasRenderingContext2D,
  lm: LM,
  scores: Record<string, number>,
  metrics: Record<string, number | string>,
  w: number, h: number,
) {
  const vis = (i: number) => (lm[i]?.visibility ?? 0) > 0.5;
  const px = (i: number) => lm[i].x * w;
  const py = (i: number) => lm[i].y * h;

  // 挙脚の判定
  const raisedLeft = lm[27].y < lm[28].y;
  const hipIdx = raisedLeft ? 23 : 24;
  const kneeIdx = raisedLeft ? 25 : 26;
  const ankleIdx = raisedLeft ? 27 : 28;
  const shoulderIdx = raisedLeft ? 11 : 12;
  const standHipIdx = raisedLeft ? 24 : 23;

  // --- 膝の伸展アーク (膝が頂点) ---
  if (vis(hipIdx) && vis(kneeIdx) && vis(ankleIdx)) {
    const kx = px(kneeIdx), ky = py(kneeIdx);
    const angToHip = Math.atan2(py(hipIdx) - ky, px(hipIdx) - kx);
    const angToAnkle = Math.atan2(py(ankleIdx) - ky, px(ankleIdx) - kx);

    const kneeScore = scores["knee_extension"] ?? 0;
    const kneeAngle = typeof metrics["knee_extension_angle"] === "number"
      ? metrics["knee_extension_angle"] : 0;

    drawArc(ctx, kx, ky, 35, angToAnkle, angToHip,
      scoreColor(kneeScore),
      `膝 ${kneeAngle.toFixed(0)}°`,
      kneeScore
    );
  }

  // --- 脚の挙上アーク (腰が頂点, 鉛直方向との角度) ---
  if (vis(hipIdx) && vis(ankleIdx)) {
    const hx = px(hipIdx), hy = py(hipIdx);
    const angToAnkle = Math.atan2(py(ankleIdx) - hy, px(ankleIdx) - hx);
    const angVertical = Math.PI / 2; // 真下

    const elevScore = scores["leg_elevation"] ?? 0;
    const elevAngle = typeof metrics["leg_elevation_angle"] === "number"
      ? metrics["leg_elevation_angle"] : 0;

    drawArc(ctx, hx, hy, 50, Math.min(angToAnkle, angVertical), Math.max(angToAnkle, angVertical),
      scoreColor(elevScore),
      `挙上 ${elevAngle.toFixed(0)}°`,
      elevScore
    );

    // プロ基準ゴーストライン (90度=水平)
    const legLen = Math.hypot(px(ankleIdx) - hx, py(ankleIdx) - hy);
    const proDirection = raisedLeft ? Math.PI : 0; // 挙脚側の水平方向
    drawProGhostLine(ctx, hx, hy, proDirection, legLen, "Pro: 90°");
  }

  // --- 背中のラインアーク (軸脚腰が頂点) ---
  if (vis(standHipIdx) && vis(shoulderIdx)) {
    const shx = px(standHipIdx), shy = py(standHipIdx);
    const angToShoulder = Math.atan2(py(shoulderIdx) - shy, px(shoulderIdx) - shx);
    const angUp = -Math.PI / 2; // 真上

    const backScore = scores["back_line"] ?? 0;
    const backAngle = typeof metrics["back_line_angle"] === "number"
      ? metrics["back_line_angle"] : 0;

    drawArc(ctx, shx, shy, 40,
      Math.min(angToShoulder, angUp),
      Math.max(angToShoulder, angUp),
      scoreColor(backScore),
      `背中 ${backAngle.toFixed(0)}°`,
      backScore
    );
  }
}

/* ================================================================
 *  パッセ用オーバーレイ
 * ================================================================ */

function drawPasseOverlays(
  ctx: CanvasRenderingContext2D,
  lm: LM,
  scores: Record<string, number>,
  metrics: Record<string, number | string>,
  w: number, h: number,
) {
  const vis = (i: number) => (lm[i]?.visibility ?? 0) > 0.5;
  const px = (i: number) => lm[i].x * w;
  const py = (i: number) => lm[i].y * h;

  const raisedLeft = lm[25].y < lm[26].y;
  const wHip = raisedLeft ? 23 : 24;
  const wKnee = raisedLeft ? 25 : 26;
  const wAnkle = raisedLeft ? 27 : 28;
  const sHip = raisedLeft ? 24 : 23;
  const sKnee = raisedLeft ? 26 : 25;
  const sAnkle = raisedLeft ? 28 : 27;

  // --- パッセ脚の膝角度アーク ---
  if (vis(wHip) && vis(wKnee) && vis(wAnkle)) {
    const kx = px(wKnee), ky = py(wKnee);
    const a1 = Math.atan2(py(wHip) - ky, px(wHip) - kx);
    const a2 = Math.atan2(py(wAnkle) - ky, px(wAnkle) - kx);

    const wkScore = scores["working_knee"] ?? 0;
    const wkAngle = typeof metrics["working_knee_angle"] === "number"
      ? metrics["working_knee_angle"] : 0;

    drawArc(ctx, kx, ky, 30, Math.min(a1, a2), Math.max(a1, a2),
      scoreColor(wkScore), `パッセ膝 ${wkAngle.toFixed(0)}°`, wkScore);
  }

  // --- 軸脚の膝伸展アーク ---
  if (vis(sHip) && vis(sKnee) && vis(sAnkle)) {
    const kx = px(sKnee), ky = py(sKnee);
    const a1 = Math.atan2(py(sHip) - ky, px(sHip) - kx);
    const a2 = Math.atan2(py(sAnkle) - ky, px(sAnkle) - kx);

    const skScore = scores["standing_knee"] ?? 0;
    const skAngle = typeof metrics["standing_knee_angle"] === "number"
      ? metrics["standing_knee_angle"] : 0;

    drawArc(ctx, kx, ky, 35, Math.min(a1, a2), Math.max(a1, a2),
      scoreColor(skScore), `軸膝 ${skAngle.toFixed(0)}°`, skScore);
  }

  // --- 骨盤水平ライン ---
  if (vis(23) && vis(24)) {
    const lhx = px(23), lhy = py(23);
    const rhx = px(24), rhy = py(24);
    const tiltScore = scores["pelvic_tilt"] ?? 0;
    const color = scoreColor(tiltScore);

    // 実際の骨盤ライン
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(lhx - 15, lhy);
    ctx.lineTo(rhx + 15, rhy);
    ctx.stroke();

    // 理想水平ライン (プロ基準)
    const midY = (lhy + rhy) / 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(34,211,238,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(lhx - 30, midY);
    ctx.lineTo(rhx + 30, midY);
    ctx.stroke();
    ctx.restore();

    drawLabel(ctx, (lhx + rhx) / 2, Math.min(lhy, rhy) - 22,
      `骨盤 ${tiltScore.toFixed(0)}`, tiltScore);
  }

  // --- アン・ドゥオール角度 ---
  if (vis(wHip) && vis(wKnee)) {
    const turnScore = scores["turnout"] ?? 0;
    const turnAngle = typeof metrics["turnout_angle"] === "number"
      ? metrics["turnout_angle"] : 0;

    drawLabel(ctx, px(wKnee) + (raisedLeft ? -30 : 30), py(wKnee) - 20,
      `T/O ${turnAngle.toFixed(0)}°`, turnScore);
  }
}

/* ================================================================
 *  ピルエット用オーバーレイ
 * ================================================================ */

function drawPirouetteOverlays(
  ctx: CanvasRenderingContext2D,
  lm: LM,
  scores: Record<string, number>,
  metrics: Record<string, number | string>,
  w: number, h: number,
) {
  const vis = (i: number) => (lm[i]?.visibility ?? 0) > 0.5;
  const px = (i: number) => lm[i].x * w;
  const py = (i: number) => lm[i].y * h;

  // パッセ脚 / 軸脚の判定
  const raisedLeft = lm[25].y < lm[26].y;
  const sHip = raisedLeft ? 24 : 23;
  const sKnee = raisedLeft ? 26 : 25;
  const sAnkle = raisedLeft ? 28 : 27;
  const sHeel = raisedLeft ? 30 : 29;
  const sFootIdx = raisedLeft ? 32 : 31;
  const wHip = raisedLeft ? 23 : 24;
  const wKnee = raisedLeft ? 25 : 26;
  const wAnkle = raisedLeft ? 27 : 28;

  // --- 1. 体幹の垂直軸ライン (肩中心→腰中心) ---
  if (vis(11) && vis(12) && vis(23) && vis(24)) {
    const scx = (px(11) + px(12)) / 2;
    const scy = (py(11) + py(12)) / 2;
    const hcx = (px(23) + px(24)) / 2;
    const hcy = (py(23) + py(24)) / 2;

    const vertScore = scores["vertical_axis"] ?? 0;
    const vertAngle = typeof metrics["vertical_axis_angle"] === "number"
      ? metrics["vertical_axis_angle"] : 0;
    const color = scoreColor(vertScore);

    // 実際の体幹ライン
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(scx, scy);
    ctx.lineTo(hcx, hcy);
    ctx.stroke();

    // 理想の垂直ライン (腰中心から真上)
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "rgba(34,211,238,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(hcx, hcy);
    ctx.lineTo(hcx, scy);
    ctx.stroke();
    ctx.restore();

    drawLabel(ctx, (scx + hcx) / 2 - 40, (scy + hcy) / 2,
      `垂直 ${vertAngle.toFixed(0)}°`, vertScore);
  }

  // --- 2. ルルヴェ指標 (軸脚の足) ---
  if (vis(sHeel) && vis(sFootIdx)) {
    const heelX = px(sHeel), heelY = py(sHeel);
    const footX = px(sFootIdx), footY = py(sFootIdx);

    const releveScore = scores["releve_height"] ?? 0;
    const color = scoreColor(releveScore);

    // ヒール→つま先ライン
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.moveTo(heelX, heelY);
    ctx.lineTo(footX, footY);
    ctx.stroke();

    // 上向き矢印 (ルルヴェ方向)
    const arrowLen = 15;
    ctx.beginPath();
    ctx.moveTo(heelX, heelY);
    ctx.lineTo(heelX, heelY - arrowLen);
    ctx.moveTo(heelX - 4, heelY - arrowLen + 4);
    ctx.lineTo(heelX, heelY - arrowLen);
    ctx.lineTo(heelX + 4, heelY - arrowLen + 4);
    ctx.stroke();
    ctx.restore();

    drawLabel(ctx, (heelX + footX) / 2, Math.min(heelY, footY) - 18,
      `ルルヴェ ${releveScore.toFixed(0)}`, releveScore);
  }

  // --- 3. 軸脚の膝伸展アーク ---
  if (vis(sHip) && vis(sKnee) && vis(sAnkle)) {
    const kx = px(sKnee), ky = py(sKnee);
    const a1 = Math.atan2(py(sHip) - ky, px(sHip) - kx);
    const a2 = Math.atan2(py(sAnkle) - ky, px(sAnkle) - kx);

    const skScore = scores["standing_knee"] ?? 0;
    const skAngle = typeof metrics["standing_knee_angle"] === "number"
      ? metrics["standing_knee_angle"] : 0;

    drawArc(ctx, kx, ky, 35, Math.min(a1, a2), Math.max(a1, a2),
      scoreColor(skScore), `軸膝 ${skAngle.toFixed(0)}°`, skScore);
  }

  // --- 4. パッセ脚の膝アーク ---
  if (vis(wHip) && vis(wKnee) && vis(wAnkle)) {
    const kx = px(wKnee), ky = py(wKnee);
    const a1 = Math.atan2(py(wHip) - ky, px(wHip) - kx);
    const a2 = Math.atan2(py(wAnkle) - ky, px(wAnkle) - kx);

    const wlScore = scores["working_leg"] ?? 0;
    const wlAngle = typeof metrics["working_knee_angle"] === "number"
      ? metrics["working_knee_angle"] : 0;

    drawArc(ctx, kx, ky, 30, Math.min(a1, a2), Math.max(a1, a2),
      scoreColor(wlScore), `パッセ膝 ${wlAngle.toFixed(0)}°`, wlScore);
  }

  // --- 5. 骨盤水平ライン ---
  if (vis(23) && vis(24)) {
    const lhx = px(23), lhy = py(23);
    const rhx = px(24), rhy = py(24);
    const plScore = scores["pelvic_level"] ?? 0;
    const color = scoreColor(plScore);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.globalAlpha = 0.8;
    ctx.beginPath();
    ctx.moveTo(lhx - 15, lhy);
    ctx.lineTo(rhx + 15, rhy);
    ctx.stroke();

    const midY = (lhy + rhy) / 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(34,211,238,0.6)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(lhx - 30, midY);
    ctx.lineTo(rhx + 30, midY);
    ctx.stroke();
    ctx.restore();

    drawLabel(ctx, (lhx + rhx) / 2, Math.min(lhy, rhy) - 22,
      `骨盤 ${plScore.toFixed(0)}`, plScore);
  }
}

/* ================================================================
 *  パ・ド・ドゥ用: デュアル骨格描画
 * ================================================================ */

function drawPairSkeletons(
  ctx: CanvasRenderingContext2D,
  pairData: PairData,
  w: number, h: number,
) {
  for (const person of pairData.persons) {
    const colorSet = PAIR_COLORS[person.person_id as 0 | 1] ?? PAIR_COLORS[0];
    const lm = person.landmarks;
    const pxFn = (i: number) => lm[i].x * w;
    const pyFn = (i: number) => lm[i].y * h;
    const visFn = (i: number) => (lm[i]?.visibility ?? 0) > 0.4;

    // 骨格接続線
    for (const [a, b] of CONNECTIONS) {
      if (!visFn(a) || !visFn(b)) continue;

      // グロー
      ctx.save();
      ctx.shadowColor = colorSet.glow;
      ctx.shadowBlur = 10;
      ctx.strokeStyle = colorSet.main;
      ctx.lineWidth = 3;
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.moveTo(pxFn(a), pyFn(a));
      ctx.lineTo(pxFn(b), pyFn(b));
      ctx.stroke();
      ctx.restore();

      // 芯線（白）
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pxFn(a), pyFn(a));
      ctx.lineTo(pxFn(b), pyFn(b));
      ctx.stroke();
    }

    // 関節点
    const majorJoints = [11, 12, 23, 24, 25, 26, 27, 28];
    for (const idx of [11, 12, 23, 24, 25, 26, 27, 28, 13, 14, 15, 16, 29, 30, 31, 32]) {
      if (!visFn(idx)) continue;
      const x = pxFn(idx);
      const y = pyFn(idx);
      const r = majorJoints.includes(idx) ? 6 : 3;

      ctx.save();
      ctx.shadowColor = colorSet.glow;
      ctx.shadowBlur = 8;
      ctx.fillStyle = colorSet.main;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
    }

    // ダンサーラベル (頭の上)
    if (visFn(0)) { // NOSE
      const noseX = pxFn(0);
      const noseY = pyFn(0);
      ctx.save();
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "center";
      ctx.fillStyle = "rgba(0,0,0,0.7)";
      const labelText = person.person_id === 0 ? "A" : "B";
      const tw = ctx.measureText(labelText).width + 12;
      ctx.beginPath();
      ctx.roundRect(noseX - tw / 2, noseY - 28, tw, 18, 9);
      ctx.fill();
      ctx.fillStyle = colorSet.main;
      ctx.fillText(labelText, noseX, noseY - 16);
      ctx.restore();
    }
  }
  ctx.globalAlpha = 1.0;
}

function drawSharedCoM(
  ctx: CanvasRenderingContext2D,
  pairData: PairData,
  _scores: Record<string, number>,
  w: number, h: number,
) {
  const pm = pairData.pair_metrics;
  const comX = pm.com_x * w;
  const comY = pm.com_y * h;
  const comScore = pairData.pair_scores?.shared_com ?? 50;

  // スコアに応じた色
  let comColor: string;
  if (comScore >= 85) comColor = "#22c55e";
  else if (comScore >= 60) comColor = "#f59e0b";
  else comColor = "#ef4444";

  // 外側グロー（大きく、半透明）
  ctx.save();
  const gradient = ctx.createRadialGradient(comX, comY, 2, comX, comY, 22);
  gradient.addColorStop(0, comColor + "cc");
  gradient.addColorStop(0.5, comColor + "44");
  gradient.addColorStop(1, comColor + "00");
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(comX, comY, 22, 0, Math.PI * 2);
  ctx.fill();

  // 中心の光る点
  ctx.shadowColor = comColor;
  ctx.shadowBlur = 15;
  ctx.fillStyle = comColor;
  ctx.beginPath();
  ctx.arc(comX, comY, 6, 0, Math.PI * 2);
  ctx.fill();

  // 白い縁
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(comX, comY, 6, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  // ラベル "CoM"
  ctx.save();
  ctx.font = "bold 10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillStyle = "rgba(0,0,0,0.7)";
  const tw = ctx.measureText("CoM").width + 10;
  ctx.beginPath();
  ctx.roundRect(comX - tw / 2, comY + 12, tw, 16, 8);
  ctx.fill();
  ctx.fillStyle = comColor;
  ctx.fillText("CoM", comX, comY + 23);
  ctx.restore();
}

function drawSupportDistanceLine(
  ctx: CanvasRenderingContext2D,
  pairData: PairData,
  _scores: Record<string, number>,
  w: number, h: number,
) {
  if (pairData.persons.length < 2) return;

  const lmA = pairData.persons[0].landmarks;
  const lmB = pairData.persons[1].landmarks;

  // Hip中心
  const hipAx = ((lmA[23]?.x ?? 0) + (lmA[24]?.x ?? 0)) / 2 * w;
  const hipAy = ((lmA[23]?.y ?? 0) + (lmA[24]?.y ?? 0)) / 2 * h;
  const hipBx = ((lmB[23]?.x ?? 0) + (lmB[24]?.x ?? 0)) / 2 * w;
  const hipBy = ((lmB[23]?.y ?? 0) + (lmB[24]?.y ?? 0)) / 2 * h;

  const distScore = pairData.pair_scores?.support_distance ?? 50;
  const color = distScore >= 85 ? "#22c55e" : distScore >= 60 ? "#f59e0b" : "#ef4444";

  // 破線
  ctx.save();
  ctx.setLineDash([6, 4]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.moveTo(hipAx, hipAy);
  ctx.lineTo(hipBx, hipBy);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();

  // 距離ラベル
  const midX = (hipAx + hipBx) / 2;
  const midY = (hipAy + hipBy) / 2;
  drawLabel(ctx, midX, midY - 16,
    `距離 ${distScore.toFixed(0)}`, distScore);
}
