"""
Shared SQLite schema and helpers for all TFG embedding databases.
Backwards-compatible with existing FakeVV qwen3vl_cache.db and wave_cache.db.
"""

import json
import sqlite3

import numpy as np


def init_db(db_path: str) -> sqlite3.Connection:
    """Create all tables (idempotent). Safe to call on existing FakeVV DBs."""
    conn = sqlite3.connect(db_path)

    # existing tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            raw_id     TEXT,
            modality   TEXT,
            vector     BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (raw_id, modality)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            raw_id     TEXT PRIMARY KEY,
            text       TEXT,
            model      TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS video_metadata (
            raw_id           TEXT PRIMARY KEY,
            duration_seconds REAL,
            has_audio        INTEGER,
            n_frames_used    INTEGER,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # migrate existing DBs: add new columns if missing (OperationalError = already exists)
    for stmt in (
        "ALTER TABLE transcripts ADD COLUMN language TEXT",
        "ALTER TABLE video_metadata ADD COLUMN dataset_origin TEXT",
    ):
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass

    # new tables
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transcript_words (
            raw_id      TEXT,
            word_idx    INTEGER,
            word        TEXT,
            start_s     REAL,
            end_s       REAL,
            probability REAL,
            PRIMARY KEY (raw_id, word_idx)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS segment_embeddings (
            raw_id       TEXT,
            segment_type TEXT,
            segment_idx  INTEGER,
            start_s      REAL,
            end_s        REAL,
            modality     TEXT,
            vector       BLOB,
            extra_json   TEXT,
            PRIMARY KEY (raw_id, segment_type, segment_idx, modality)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scene_metadata (
            raw_id     TEXT,
            scene_idx  INTEGER,
            start_s    REAL,
            end_s      REAL,
            detector   TEXT,
            confidence REAL,
            PRIMARY KEY (raw_id, scene_idx)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groundlie_bboxes (
            raw_id   TEXT,
            frame_id INTEGER,
            x        REAL,
            y        REAL,
            w        REAL,
            h        REAL,
            PRIMARY KEY (raw_id, frame_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS groundlie_labels (
            raw_id        TEXT PRIMARY KEY,
            binary_label  INTEGER,
            false_title   INTEGER,
            false_speech  INTEGER,
            temporal_edit INTEGER,
            cgi           INTEGER,
            contradictory INTEGER,
            unsupported   INTEGER,
            title         TEXT,
            title_fake    TEXT
        )
    """)

    # M3A-specific tables (Xu et al. 2024). MM/MTG mappings live in the index JSON.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS m3a_meta (
            raw_id           TEXT PRIMARY KEY,
            outlet           TEXT,
            topic            TEXT,
            sentiment        TEXT,
            geography        TEXT,
            duration_seconds REAL,
            summary          TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS m3a_nem (
            raw_id   TEXT,
            subtype  TEXT,
            text     TEXT,
            PRIMARY KEY (raw_id, subtype)
        )
    """)

    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Embedding helpers (identical signatures to existing scripts)
# ---------------------------------------------------------------------------

def save_embedding(conn: sqlite3.Connection, raw_id: str, modality: str, emb) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO embeddings (raw_id, modality, vector) VALUES (?, ?, ?)",
        (raw_id, modality, emb.astype(np.float32).tobytes()),
    )
    conn.commit()


def load_embedding(conn: sqlite3.Connection, raw_id: str, modality: str):
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE raw_id=? AND modality=?",
        (raw_id, modality),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def cache_progress(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT modality, COUNT(*) FROM embeddings GROUP BY modality"
    ).fetchall()
    return {mod: cnt for mod, cnt in rows}


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def save_transcript_text(
    conn: sqlite3.Connection,
    raw_id: str,
    text: str,
    model_name: str = "whisper-small",
    language: str = None,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO transcripts (raw_id, text, model, language) VALUES (?, ?, ?, ?)",
        (raw_id, text, model_name, language),
    )
    conn.commit()


def load_transcript_text(conn: sqlite3.Connection, raw_id: str):
    row = conn.execute("SELECT text FROM transcripts WHERE raw_id=?", (raw_id,)).fetchone()
    return row[0] if row else None


def save_transcript_words(conn: sqlite3.Connection, raw_id: str, words: list) -> None:
    """words: list of dicts with keys word, start_s, end_s, probability."""
    conn.executemany(
        """INSERT OR IGNORE INTO transcript_words
           (raw_id, word_idx, word, start_s, end_s, probability)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (raw_id, i, w["word"], w.get("start_s"), w.get("end_s"), w.get("probability"))
            for i, w in enumerate(words)
        ],
    )
    conn.commit()


def load_transcript_words(conn: sqlite3.Connection, raw_id: str) -> list:
    rows = conn.execute(
        "SELECT word, start_s, end_s, probability FROM transcript_words "
        "WHERE raw_id=? ORDER BY word_idx",
        (raw_id,),
    ).fetchall()
    return [{"word": r[0], "start_s": r[1], "end_s": r[2], "probability": r[3]} for r in rows]


# ---------------------------------------------------------------------------
# Video metadata helpers (identical signatures to existing scripts)
# ---------------------------------------------------------------------------

def save_video_metadata(
    conn: sqlite3.Connection,
    raw_id: str,
    duration_seconds,
    has_audio: bool,
    n_frames_used: int = None,
    dataset_origin: str = None,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO video_metadata
           (raw_id, duration_seconds, has_audio, n_frames_used, dataset_origin)
           VALUES (?, ?, ?, ?, ?)""",
        (raw_id, duration_seconds, int(has_audio), n_frames_used, dataset_origin),
    )
    conn.commit()


def load_video_metadata(conn: sqlite3.Connection, raw_id: str):
    row = conn.execute(
        "SELECT duration_seconds, has_audio, n_frames_used FROM video_metadata WHERE raw_id=?",
        (raw_id,),
    ).fetchone()
    if row is None:
        return None
    return {"duration_seconds": row[0], "has_audio": bool(row[1]), "n_frames_used": row[2]}


# ---------------------------------------------------------------------------
# Scene metadata helpers
# ---------------------------------------------------------------------------

def save_scene_metadata(conn: sqlite3.Connection, raw_id: str, scenes: list) -> None:
    """scenes: list of dicts with keys scene_idx, start_s, end_s, detector, confidence."""
    conn.executemany(
        """INSERT OR IGNORE INTO scene_metadata
           (raw_id, scene_idx, start_s, end_s, detector, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (raw_id, s["scene_idx"], s["start_s"], s["end_s"],
             s.get("detector"), s.get("confidence"))
            for s in scenes
        ],
    )
    conn.commit()


def load_scene_metadata(conn: sqlite3.Connection, raw_id: str) -> list:
    rows = conn.execute(
        "SELECT scene_idx, start_s, end_s, detector, confidence FROM scene_metadata "
        "WHERE raw_id=? ORDER BY scene_idx",
        (raw_id,),
    ).fetchall()
    return [
        {"scene_idx": r[0], "start_s": r[1], "end_s": r[2],
         "detector": r[3], "confidence": r[4]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Segment embedding helpers
# ---------------------------------------------------------------------------

def save_segment_embedding(
    conn: sqlite3.Connection,
    raw_id: str,
    segment_type: str,
    segment_idx: int,
    start_s: float,
    end_s: float,
    modality: str,
    vector,
    extra_json: dict = None,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO segment_embeddings
           (raw_id, segment_type, segment_idx, start_s, end_s, modality, vector, extra_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            raw_id, segment_type, segment_idx, start_s, end_s, modality,
            vector.astype(np.float32).tobytes(),
            json.dumps(extra_json) if extra_json is not None else None,
        ),
    )
    conn.commit()


def load_segment_embeddings(
    conn: sqlite3.Connection,
    raw_id: str,
    segment_type: str,
    modality: str,
) -> list:
    rows = conn.execute(
        """SELECT segment_idx, start_s, end_s, vector, extra_json
           FROM segment_embeddings
           WHERE raw_id=? AND segment_type=? AND modality=?
           ORDER BY segment_idx""",
        (raw_id, segment_type, modality),
    ).fetchall()
    return [
        {
            "segment_idx": r[0],
            "start_s": r[1],
            "end_s": r[2],
            "vector": np.frombuffer(r[3], dtype=np.float32),
            "extra": json.loads(r[4]) if r[4] else None,
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# GroundLie360-specific helpers
# ---------------------------------------------------------------------------

def save_groundlie_labels(conn: sqlite3.Connection, raw_id: str, entry: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO groundlie_labels
           (raw_id, binary_label, false_title, false_speech, temporal_edit,
            cgi, contradictory, unsupported, title, title_fake)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            raw_id,
            entry.get("binary_label"),
            entry.get("false_title"),
            entry.get("false_speech"),
            entry.get("temporal_edit"),
            entry.get("cgi"),
            entry.get("contradictory"),
            entry.get("unsupported"),
            entry.get("title"),
            entry.get("title_fake"),
        ),
    )
    conn.commit()


def save_groundlie_bboxes(conn: sqlite3.Connection, raw_id: str, bboxes: list) -> None:
    """bboxes: list of dicts with keys frame_id, bbox=[x,y,w,h]."""
    conn.executemany(
        """INSERT OR IGNORE INTO groundlie_bboxes
           (raw_id, frame_id, x, y, w, h)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (raw_id, b["frame_id"], b["bbox"][0], b["bbox"][1], b["bbox"][2], b["bbox"][3])
            for b in bboxes
        ],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# M3A-specific helpers
# ---------------------------------------------------------------------------

def save_m3a_meta(conn: sqlite3.Connection, raw_id: str, entry: dict) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO m3a_meta
           (raw_id, outlet, topic, sentiment, geography, duration_seconds, summary)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            raw_id,
            entry.get("outlet"),
            entry.get("topic"),
            entry.get("sentiment"),
            entry.get("geography"),
            entry.get("duration_seconds"),
            entry.get("summary"),
        ),
    )
    conn.commit()


def save_m3a_nem(conn: sqlite3.Connection, raw_id: str, nem_texts: dict) -> None:
    """nem_texts: {subtype: text} for the present NEM subtypes."""
    conn.executemany(
        "INSERT OR IGNORE INTO m3a_nem (raw_id, subtype, text) VALUES (?, ?, ?)",
        [(raw_id, sub, txt) for sub, txt in nem_texts.items() if txt],
    )
    conn.commit()
