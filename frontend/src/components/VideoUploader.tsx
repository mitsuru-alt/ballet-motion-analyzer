import { useCallback, useRef, useState } from "react";

interface Props {
  onFileSelect: (file: File) => void;
  loading: boolean;
}

export default function VideoUploader({ onFileSelect, loading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [hover, setHover] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (file.type.startsWith("image/")) {
        setPreview(URL.createObjectURL(file));
      } else {
        setPreview(null);
      }
      onFileSelect(file);
    },
    [onFileSelect]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const isActive = dragOver || hover;

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
      onClick={() => inputRef.current?.click()}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        border: `2px dashed ${dragOver ? "#6366f1" : isActive ? "rgba(165,180,252,0.6)" : "rgba(203,213,225,0.8)"}`,
        borderRadius: 20,
        padding: "48px 40px",
        textAlign: "center",
        cursor: loading ? "wait" : "pointer",
        background: dragOver
          ? "linear-gradient(135deg, rgba(238,242,255,0.8), rgba(224,231,255,0.6))"
          : isActive
            ? "rgba(255,255,255,0.8)"
            : "rgba(255,255,255,0.6)",
        transition: "all 0.3s cubic-bezier(0.4,0,0.2,1)",
        backdropFilter: "blur(8px)",
        boxShadow: dragOver
          ? "0 8px 32px rgba(99,102,241,0.15), inset 0 0 0 1px rgba(99,102,241,0.1)"
          : isActive
            ? "0 4px 20px rgba(99,102,241,0.08)"
            : "0 2px 12px rgba(0,0,0,0.04)",
        transform: isActive && !loading ? "translateY(-2px)" : "none",
        position: "relative",
        overflow: "hidden",
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,video/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) handleFile(file);
        }}
      />

      {loading ? (
        <div>
          <div
            style={{
              width: 40,
              height: 40,
              border: "3px solid rgba(99,102,241,0.15)",
              borderTopColor: "#6366f1",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
              margin: "0 auto 12px",
            }}
          />
          <p style={{ color: "#6366f1", fontWeight: 600, fontSize: 14 }}>
            解析中...
          </p>
        </div>
      ) : preview ? (
        <div>
          <img
            src={preview}
            alt="preview"
            style={{
              maxHeight: 200,
              borderRadius: 12,
              marginBottom: 12,
              boxShadow: "0 4px 16px rgba(0,0,0,0.1)",
            }}
          />
          <p style={{ color: "#6b7280", fontSize: 13 }}>
            クリックして別のファイルを選択
          </p>
        </div>
      ) : (
        <div>
          {/* Upload icon with float animation */}
          <div
            style={{
              width: 64,
              height: 64,
              borderRadius: 20,
              background: "linear-gradient(135deg, #eef2ff, #e0e7ff)",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              marginBottom: 16,
              animation: "float 3s ease-in-out infinite",
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
              strokeLinejoin="round"
            >
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p
            style={{
              fontSize: 17,
              color: "#1e293b",
              marginBottom: 8,
              fontWeight: 600,
              letterSpacing: "-0.01em",
            }}
          >
            画像または動画をドラッグ&ドロップ
          </p>
          <p style={{ color: "#94a3b8", fontSize: 13, lineHeight: 1.5 }}>
            JPEG, PNG, MP4 対応 / 全身が映るように撮影してください
          </p>
          <div
            style={{
              marginTop: 20,
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "9px 22px",
              borderRadius: 12,
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              color: "#fff",
              fontSize: 13,
              fontWeight: 600,
              boxShadow: "0 2px 8px rgba(99,102,241,0.3)",
              transition: "all 0.2s ease",
            }}
          >
            ファイルを選択
          </div>
        </div>
      )}
    </div>
  );
}
