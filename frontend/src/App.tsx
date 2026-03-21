import { useState, useEffect, useRef } from "react";
import VideoUploader from "./components/VideoUploader";
import SkeletonOverlay from "./components/SkeletonOverlay";
import MetricsPanel from "./components/MetricsPanel";
import AdviceCard from "./components/AdviceCard";
import RotationPanel from "./components/RotationPanel";
// PddPanel removed (pirouette-only app)
import Dashboard from "./components/Dashboard";
import { useAnalysis } from "./hooks/useAnalysis";
import { useHistory } from "./hooks/useHistory";

function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  const frac = Math.floor((ms % 1000) / 100);
  return `${min}:${String(sec).padStart(2, "0")}.${frac}`;
}

export default function App() {
  const { result, loading, error, analyze, reset, uploadProgress, phase } =
    useAnalysis();
  const {
    records,
    loading: historyLoading,
    fetchHistory,
    saveToHistory,
    generateHighlights,
  } = useHistory();
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [headerBtnHover, setHeaderBtnHover] = useState(false);
  const [metricsHover, setMetricsHover] = useState(false);
  const [rotationHover, setRotationHover] = useState(false);
  const [adviceHover, setAdviceHover] = useState(false);
  const [view, setView] = useState<"analyze" | "dashboard">("analyze");
  const [dashBtnHover, setDashBtnHover] = useState(false);
  const savedResultRef = useRef<string | null>(null);

  // Responsive: detect narrow screens
  const [isMobile, setIsMobile] = useState(window.innerWidth < 900);
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 900);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // 解析結果が出たら自動保存
  useEffect(() => {
    if (result && result.overall_score > 0) {
      const resultKey = `${result.pose_type}_${result.overall_score}_${Date.now()}`;
      if (savedResultRef.current !== resultKey) {
        savedResultRef.current = resultKey;
        saveToHistory({
          pose_type: result.pose_type,
          overall_score: result.overall_score,
          scores: result.scores,
          metrics: result.metrics,
          advice: result.advice,
          rotation_data: result.rotation_data,
          // pair_data removed
        });
      }
    }
  }, [result, saveToHistory]);

  // ダッシュボード表示時に履歴をフェッチ
  useEffect(() => {
    if (view === "dashboard") {
      fetchHistory();
    }
  }, [view, fetchHistory]);

  const handleFileSelect = (file: File) => {
    if (file.type.startsWith("image/")) {
      setImageSrc(URL.createObjectURL(file));
      setIsVideo(false);
    } else {
      setImageSrc(null);
      setIsVideo(true);
    }
    analyze(file);
  };

  const handleReset = () => {
    reset();
    setImageSrc(null);
    setIsVideo(false);
    savedResultRef.current = null;
  };

  const overlayImageSrc =
    imageSrc ||
    (result?.best_frame_image
      ? `data:image/jpeg;base64,${result.best_frame_image}`
      : null);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(135deg, #faf5ff 0%, #f0f4ff 40%, #f8fafc 100%)",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif',
        position: "relative",
        overflow: "hidden",
      }}
    >
      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px) } to { opacity: 1; transform: translateY(0) } }
        @keyframes fadeInScale { from { opacity: 0; transform: scale(0.95) translateY(12px) } to { opacity: 1; transform: scale(1) translateY(0) } }
        @keyframes progressPulse { 0%,100% { opacity: 1 } 50% { opacity: 0.7 } }
        @keyframes shimmer { 0% { background-position: -200% 0 } 100% { background-position: 200% 0 } }
        @keyframes float { 0%,100% { transform: translateY(0) } 50% { transform: translateY(-6px) } }
        @keyframes slideInRight { from { opacity: 0; transform: translateX(20px) } to { opacity: 1; transform: translateX(0) } }
        @keyframes ringDraw { from { stroke-dashoffset: 339.292 } }
        @keyframes barGrow { from { width: 0% } }
        * { box-sizing: border-box; margin: 0; padding: 0; }
      `}</style>

      {/* Decorative background orbs */}
      <div
        style={{
          position: "fixed",
          top: -200,
          right: -200,
          width: 600,
          height: 600,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(139,92,246,0.08) 0%, rgba(99,102,241,0.04) 40%, transparent 70%)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />
      <div
        style={{
          position: "fixed",
          bottom: -300,
          left: -200,
          width: 500,
          height: 500,
          borderRadius: "50%",
          background:
            "radial-gradient(circle, rgba(236,72,153,0.06) 0%, transparent 60%)",
          pointerEvents: "none",
          zIndex: 0,
        }}
      />

      {/* Header */}
      <header
        style={{
          padding: "12px 32px",
          background: "rgba(255,255,255,0.72)",
          backdropFilter: "blur(20px) saturate(180%)",
          WebkitBackdropFilter: "blur(20px) saturate(180%)",
          borderBottom: "1px solid rgba(226,232,240,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          position: "sticky",
          top: 0,
          zIndex: 10,
          boxShadow: "0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.02)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: 10,
              background: "linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontSize: 16,
              fontWeight: 800,
              boxShadow: "0 2px 8px rgba(99,102,241,0.35)",
              transition: "transform 0.2s ease, box-shadow 0.2s ease",
            }}
          >
            B
          </div>
          <h1
            style={{
              fontSize: 18,
              fontWeight: 700,
              color: "#0f172a",
              letterSpacing: "-0.02em",
            }}
          >
            Pirouette Analyzer
          </h1>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={() => setView(view === "dashboard" ? "analyze" : "dashboard")}
            onMouseOver={() => setDashBtnHover(true)}
            onMouseOut={() => setDashBtnHover(false)}
            style={{
              padding: "8px 20px",
              background: view === "dashboard"
                ? "linear-gradient(135deg, #6366f1, #8b5cf6)"
                : dashBtnHover ? "#f8f9ff" : "rgba(255,255,255,0.9)",
              border: `1px solid ${view === "dashboard" ? "#6366f1" : dashBtnHover ? "#a5b4fc" : "rgba(209,213,219,0.6)"}`,
              borderRadius: 10,
              cursor: "pointer",
              fontSize: 13,
              fontWeight: 600,
              color: view === "dashboard" ? "#fff" : dashBtnHover ? "#4338ca" : "#374151",
              transition: "all 0.2s cubic-bezier(0.4,0,0.2,1)",
              boxShadow: dashBtnHover || view === "dashboard"
                ? "0 2px 8px rgba(99,102,241,0.2)"
                : "0 1px 2px rgba(0,0,0,0.05)",
              backdropFilter: "blur(8px)",
              transform: dashBtnHover ? "translateY(-1px)" : "none",
            }}
          >
            📊 成長記録
          </button>
          {result && (
            <button
              onClick={handleReset}
              onMouseOver={() => setHeaderBtnHover(true)}
              onMouseOut={() => setHeaderBtnHover(false)}
              style={{
                padding: "8px 20px",
                background: headerBtnHover ? "#f8f9ff" : "rgba(255,255,255,0.9)",
                border: `1px solid ${headerBtnHover ? "#a5b4fc" : "rgba(209,213,219,0.6)"}`,
                borderRadius: 10,
                cursor: "pointer",
                fontSize: 13,
                fontWeight: 600,
                color: headerBtnHover ? "#4338ca" : "#374151",
                transition: "all 0.2s cubic-bezier(0.4,0,0.2,1)",
                boxShadow: headerBtnHover
                  ? "0 2px 8px rgba(99,102,241,0.12)"
                  : "0 1px 2px rgba(0,0,0,0.05)",
                backdropFilter: "blur(8px)",
                transform: headerBtnHover ? "translateY(-1px)" : "none",
              }}
            >
              新しい解析
            </button>
          )}
        </div>
      </header>

      <main
        style={{
          maxWidth: 1140,
          margin: "0 auto",
          padding: "24px 20px",
          position: "relative",
          zIndex: 1,
        }}
      >
        {/* ===== Dashboard View ===== */}
        {view === "dashboard" && (
          <Dashboard
            records={records}
            highlights={generateHighlights()}
            loading={historyLoading}
            onBack={() => setView("analyze")}
          />
        )}

        {/* ===== Upload Screen ===== */}
        {view === "analyze" && !result && !loading && !error && (
          <div
            style={{
              maxWidth: 580,
              margin: "60px auto",
              animation: "fadeIn 0.5s ease",
            }}
          >
            <div style={{ textAlign: "center", marginBottom: 32 }}>
              {/* Ballet icon */}
              <div style={{ marginBottom: 16 }}>
                <div
                  style={{
                    width: 56,
                    height: 56,
                    borderRadius: 16,
                    background:
                      "linear-gradient(135deg, #eef2ff 0%, #e0e7ff 100%)",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    boxShadow: "0 4px 12px rgba(99,102,241,0.1)",
                  }}
                >
                  <svg
                    width="28"
                    height="28"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#6366f1"
                    strokeWidth="2"
                    strokeLinecap="round"
                  >
                    <circle cx="12" cy="5" r="3" />
                    <line x1="12" y1="8" x2="12" y2="16" />
                    <line x1="12" y1="16" x2="8" y2="22" />
                    <line x1="12" y1="16" x2="16" y2="22" />
                    <line x1="12" y1="11" x2="8" y2="8" />
                    <line x1="12" y1="11" x2="16" y2="14" />
                  </svg>
                </div>
              </div>
              <h2
                style={{
                  fontSize: 28,
                  fontWeight: 800,
                  color: "#0f172a",
                  marginBottom: 10,
                  letterSpacing: "-0.03em",
                  lineHeight: 1.2,
                }}
              >
                回転を解析
              </h2>
              <p
                style={{
                  color: "#64748b",
                  fontSize: 15,
                  lineHeight: 1.7,
                  maxWidth: 440,
                  margin: "0 auto",
                }}
              >
                練習動画または写真をアップロードすると、AIが骨格検知を行い
                <br />
                プロの基準と比較した改善アドバイスを提示します
              </p>
            </div>
            <VideoUploader onFileSelect={handleFileSelect} loading={loading} />
            <div
              style={{
                display: "flex",
                gap: 10,
                justifyContent: "center",
                marginTop: 24,
                flexWrap: "wrap",
              }}
            >
              {["ピルエット専門", "回転カウント", "フォーム分析", "JPEG", "PNG", "MP4"].map((tag) => (
                <span
                  key={tag}
                  style={{
                    padding: "4px 14px",
                    borderRadius: 20,
                    background: "rgba(255,255,255,0.7)",
                    border: "1px solid rgba(226,232,240,0.8)",
                    fontSize: 11,
                    color: "#64748b",
                    fontWeight: 500,
                    backdropFilter: "blur(4px)",
                  }}
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* ===== Loading ===== */}
        {view === "analyze" && loading && (
          <div
            style={{
              textAlign: "center",
              padding: 100,
              animation: "fadeIn 0.3s ease",
            }}
          >
            {/* Double-ring spinner */}
            <div
              style={{
                width: 56,
                height: 56,
                border: "3px solid rgba(99,102,241,0.1)",
                borderTopColor: "#6366f1",
                borderRadius: "50%",
                animation: "spin 0.8s cubic-bezier(0.4,0,0.2,1) infinite",
                margin: "0 auto 24px",
                position: "relative",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  top: 6,
                  left: 6,
                  right: 6,
                  bottom: 6,
                  border: "3px solid rgba(139,92,246,0.1)",
                  borderBottomColor: "#8b5cf6",
                  borderRadius: "50%",
                  animation: "spin 1.2s cubic-bezier(0.4,0,0.2,1) infinite reverse",
                }}
              />
            </div>
            <p style={{ color: "#1e293b", fontWeight: 600, fontSize: 15 }}>
              {phase === "uploading"
                ? "ファイルをアップロード中..."
                : "MediaPipeで骨格を検出中..."}
            </p>
            <p style={{ color: "#94a3b8", fontSize: 13, marginTop: 6 }}>
              {isVideo
                ? "動画の解析には数十秒かかることがあります"
                : "画像サイズにより数秒かかることがあります"}
            </p>

            {/* Progress Bar */}
            <div
              style={{
                maxWidth: 360,
                margin: "24px auto 0",
                height: 6,
                background: "rgba(226,232,240,0.6)",
                borderRadius: 6,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width:
                    phase === "analyzing" ? "100%" : `${uploadProgress}%`,
                  height: "100%",
                  background:
                    "linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa, #8b5cf6, #6366f1)",
                  backgroundSize: "200% 100%",
                  borderRadius: 6,
                  transition: "width 0.4s cubic-bezier(0.4,0,0.2,1)",
                  animation:
                    phase === "analyzing"
                      ? "shimmer 2s linear infinite, progressPulse 1.5s ease-in-out infinite"
                      : "none",
                }}
              />
            </div>
            <p style={{ color: "#94a3b8", fontSize: 11, marginTop: 8 }}>
              {phase === "uploading" ? `${uploadProgress}%` : "解析処理中..."}
            </p>
          </div>
        )}

        {/* ===== Error ===== */}
        {view === "analyze" && error && (
          <div
            style={{
              maxWidth: 520,
              margin: "60px auto",
              padding: 32,
              background: "rgba(255,255,255,0.9)",
              border: "1px solid rgba(254,202,202,0.6)",
              borderTop: "3px solid #ef4444",
              borderRadius: 20,
              textAlign: "center",
              animation: "fadeInScale 0.3s ease",
              boxShadow: "0 4px 24px rgba(220,38,38,0.08)",
              backdropFilter: "blur(12px)",
            }}
          >
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 14,
                background: "#fef2f2",
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: 22,
                marginBottom: 14,
                color: "#dc2626",
                fontWeight: 700,
              }}
            >
              !
            </div>
            <p
              style={{
                fontWeight: 600,
                color: "#991b1b",
                fontSize: 16,
                marginBottom: 6,
              }}
            >
              解析エラー
            </p>
            <p
              style={{
                fontSize: 14,
                color: "#6b7280",
                marginBottom: 20,
                lineHeight: 1.5,
              }}
            >
              {error}
            </p>
            <button
              onClick={handleReset}
              style={{
                padding: "10px 28px",
                background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
                color: "#fff",
                border: "none",
                borderRadius: 12,
                cursor: "pointer",
                fontWeight: 600,
                fontSize: 14,
                boxShadow: "0 2px 8px rgba(99,102,241,0.3)",
                transition: "all 0.2s cubic-bezier(0.4,0,0.2,1)",
              }}
            >
              やり直す
            </button>
          </div>
        )}

        {/* ===== Results ===== */}
        {view === "analyze" && result && (
          <div style={{ animation: "fadeIn 0.4s ease" }}>
            {/* Video timestamp info */}
            {result.video_duration_ms > 0 && (
              <div
                style={{
                  marginBottom: 20,
                  padding: "12px 20px",
                  background:
                    "linear-gradient(135deg, rgba(238,242,255,0.8), rgba(224,231,255,0.6))",
                  borderRadius: 14,
                  display: "flex",
                  gap: 24,
                  alignItems: "center",
                  fontSize: 13,
                  color: "#4338ca",
                  fontWeight: 500,
                  borderLeft: "3px solid #6366f1",
                  backdropFilter: "blur(8px)",
                  boxShadow: "0 2px 8px rgba(99,102,241,0.06)",
                }}
              >
                <span>
                  動画解析: {formatTimestamp(result.video_duration_ms)} /{" "}
                  {result.total_frames_analyzed}フレーム解析
                </span>
                <span>
                  ベストフレーム:{" "}
                  {formatTimestamp(result.best_frame_timestamp_ms)} (フレーム #
                  {result.best_frame_index})
                </span>
              </div>
            )}

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 360px",
                gap: 24,
                alignItems: "start",
              }}
            >
              {/* Left: skeleton + re-upload */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 16,
                }}
              >
                {overlayImageSrc && result.landmarks.length > 0 ? (
                  <>
                    <SkeletonOverlay
                      imageSrc={overlayImageSrc}
                      frameData={result.landmarks[0]}
                      scores={result.scores}
                      metrics={result.metrics}
                      poseType={result.pose_type}
                      width={700}
                    />
                    {result.best_frame_timestamp_ms > 0 && (
                      <div
                        style={{
                          textAlign: "center",
                          fontSize: 12,
                          color: "#64748b",
                          marginTop: -8,
                        }}
                      >
                        解析フレーム:{" "}
                        {formatTimestamp(result.best_frame_timestamp_ms)}
                      </div>
                    )}
                  </>
                ) : (
                  <div
                    style={{
                      padding: 60,
                      background: "rgba(255,255,255,0.8)",
                      borderRadius: 20,
                      textAlign: "center",
                      color: "#64748b",
                      border: "1px solid rgba(226,232,240,0.6)",
                      backdropFilter: "blur(12px)",
                    }}
                  >
                    <p style={{ fontSize: 15, fontWeight: 500 }}>解析完了</p>
                    <p
                      style={{
                        fontSize: 13,
                        color: "#94a3b8",
                        marginTop: 4,
                      }}
                    >
                      ポーズのスコアと改善アドバイスを右側に表示しています
                    </p>
                  </div>
                )}

                <VideoUploader
                  onFileSelect={handleFileSelect}
                  loading={loading}
                />
              </div>

              {/* Right: scores only */}
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 20,
                }}
              >
                <div
                  onMouseEnter={() => setMetricsHover(true)}
                  onMouseLeave={() => setMetricsHover(false)}
                  style={{
                    background: "rgba(255,255,255,0.8)",
                    borderRadius: 20,
                    padding: 24,
                    border: "1px solid rgba(226,232,240,0.6)",
                    boxShadow: metricsHover
                      ? "0 12px 40px rgba(0,0,0,0.08)"
                      : "0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06)",
                    backdropFilter: "blur(12px)",
                    transition:
                      "box-shadow 0.3s ease, transform 0.3s ease",
                    transform: metricsHover ? "translateY(-2px)" : "none",
                    animation:
                      "fadeInScale 0.5s cubic-bezier(0.4,0,0.2,1)",
                  }}
                >
                  <MetricsPanel data={result} />
                </div>
              </div>
            </div>

            {/* Full-width: Rotation + Advice side by side */}
            {(result.rotation_data || result.advice.length > 0) && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns:
                    result.rotation_data && result.advice.length > 0 && !isMobile
                      ? "1fr 1fr"
                      : "1fr",
                  gap: 24,
                  marginTop: 24,
                }}
              >
                {result.rotation_data && (
                  <div
                    onMouseEnter={() => setRotationHover(true)}
                    onMouseLeave={() => setRotationHover(false)}
                    style={{
                      background: "rgba(255,255,255,0.8)",
                      borderRadius: 20,
                      padding: 24,
                      border: "1px solid rgba(226,232,240,0.6)",
                      boxShadow: rotationHover
                        ? "0 12px 40px rgba(0,0,0,0.08)"
                        : "0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06)",
                      backdropFilter: "blur(12px)",
                      transition:
                        "box-shadow 0.3s ease, transform 0.3s ease",
                      transform: rotationHover ? "translateY(-2px)" : "none",
                      animation:
                        "fadeInScale 0.5s cubic-bezier(0.4,0,0.2,1) 0.1s both",
                    }}
                  >
                    <RotationPanel data={result.rotation_data} />
                  </div>
                )}
                {result.advice.length > 0 && (
                  <div
                    onMouseEnter={() => setAdviceHover(true)}
                    onMouseLeave={() => setAdviceHover(false)}
                    style={{
                      background: "rgba(255,255,255,0.8)",
                      borderRadius: 20,
                      padding: 24,
                      border: "1px solid rgba(226,232,240,0.6)",
                      boxShadow: adviceHover
                        ? "0 12px 40px rgba(0,0,0,0.08)"
                        : "0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06)",
                      backdropFilter: "blur(12px)",
                      transition:
                        "box-shadow 0.3s ease, transform 0.3s ease",
                      transform: adviceHover ? "translateY(-2px)" : "none",
                      animation:
                        "fadeInScale 0.5s cubic-bezier(0.4,0,0.2,1) 0.15s both",
                    }}
                  >
                    <AdviceCard advice={result.advice} />
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
