"""
履歴データベース (SQLite) — 解析結果の保存・取得

テーブル:
  analysis_history
    id              INTEGER PRIMARY KEY
    created_at      TEXT (ISO 8601)
    pose_type       TEXT
    overall_score   REAL
    scores_json     TEXT (JSON)
    metrics_json    TEXT (JSON)
    advice_json     TEXT (JSON)
    rotation_json   TEXT (JSON or NULL)
    pair_json       TEXT (JSON or NULL)
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_DIR = Path(__file__).resolve().parent.parent / "data"
_DB_PATH = _DB_DIR / "history.db"

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS analysis_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    pose_type       TEXT    NOT NULL,
    overall_score   REAL    NOT NULL,
    scores_json     TEXT    NOT NULL,
    metrics_json    TEXT    NOT NULL,
    advice_json     TEXT    NOT NULL,
    rotation_json   TEXT,
    pair_json       TEXT
);
"""


def _get_conn() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def save_analysis(data: Dict[str, Any]) -> int:
    """解析結果を保存し、挿入されたIDを返す"""
    conn = _get_conn()
    try:
        cursor = conn.execute(
            """
            INSERT INTO analysis_history
                (created_at, pose_type, overall_score,
                 scores_json, metrics_json, advice_json,
                 rotation_json, pair_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                data.get("pose_type", "unknown"),
                data.get("overall_score", 0),
                json.dumps(data.get("scores", {}), ensure_ascii=False),
                json.dumps(data.get("metrics", {}), ensure_ascii=False),
                json.dumps(data.get("advice", []), ensure_ascii=False),
                json.dumps(data.get("rotation_data")) if data.get("rotation_data") else None,
                json.dumps(data.get("pair_data")) if data.get("pair_data") else None,
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def get_history(
    limit: int = 100,
    offset: int = 0,
    pose_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """保存された解析履歴を取得（新しい順）"""
    conn = _get_conn()
    try:
        query = "SELECT * FROM analysis_history"
        params: List[Any] = []
        if pose_type:
            query += " WHERE pose_type = ?"
            params.append(pose_type)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]
    finally:
        conn.close()


def get_history_count(pose_type: Optional[str] = None) -> int:
    """履歴の件数を返す"""
    conn = _get_conn()
    try:
        query = "SELECT COUNT(*) FROM analysis_history"
        params: List[Any] = []
        if pose_type:
            query += " WHERE pose_type = ?"
            params.append(pose_type)
        return conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()


def delete_record(record_id: int) -> bool:
    """レコードを削除"""
    conn = _get_conn()
    try:
        cursor = conn.execute("DELETE FROM analysis_history WHERE id = ?", (record_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["scores"] = json.loads(d.pop("scores_json"))
    d["metrics"] = json.loads(d.pop("metrics_json"))
    d["advice"] = json.loads(d.pop("advice_json"))
    rot = d.pop("rotation_json")
    d["rotation_data"] = json.loads(rot) if rot else None
    pair = d.pop("pair_json")
    d["pair_data"] = json.loads(pair) if pair else None
    return d
