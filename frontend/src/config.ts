/**
 * API Base URL
 *
 * - 開発時: Vite proxy 経由で /api → localhost:8000 にルーティング
 * - 本番時: VITE_API_URL 環境変数で外部APIサーバーを指定
 *
 * 例: VITE_API_URL=https://ballet-api.onrender.com
 */
export const API_BASE = import.meta.env.VITE_API_URL || "";
