# ==================================================
# db_logger.py  —  Database Logging Module
# Compatible: Supabase (PostgreSQL) / SQLite fallback
#
# ENV yang dibutuhkan:
#   DATABASE_URL = postgresql://user:pass@host:port/db
#   (kosongkan → otomatis pakai SQLite lokal)
#
# Tabel yang dibuat otomatis:
#   signals       — setiap signal yang di-generate
#   div_alerts    — RSI divergence alerts
#   score_history — per-kategori score untuk ML training
#   outcomes      — hasil trade (TP hit / SL hit) — isi manual / webhook
# ==================================================

import os
import time
import json
import datetime as dt
import sqlite3
import threading
from typing import Optional

# ── env ──────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ── tentukan backend ──────────────────────────────
# PostgreSQL jika DATABASE_URL tersedia, SQLite jika tidak
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgresql"))

if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        print("✅ DB: PostgreSQL (Supabase) mode")
    except ImportError:
        USE_POSTGRES = False
        print("⚠️  psycopg2 tidak terinstall → fallback ke SQLite")
else:
    print("ℹ️  DATABASE_URL kosong → SQLite mode (data tidak persist di Railway free)")

SQLITE_PATH = os.getenv("SQLITE_PATH", "trading_bot.db")

# ── thread-local connection (SQLite tidak thread-safe) ───
_local = threading.local()

# ==================================================
# CONNECTION
# ==================================================
def _get_conn():
    """Ambil koneksi DB. Thread-safe untuk SQLite."""
    if USE_POSTGRES:
        # PostgreSQL: buat koneksi baru per call (connection pool ideal tapi overkill)
        conn = psycopg2.connect(DATABASE_URL)
        conn.autocommit = False
        return conn
    else:
        # SQLite: satu koneksi per thread
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
        return _local.conn

def _execute(sql: str, params=None, fetch: bool = False):
    """
    Helper execute — handle perbedaan placeholder
    PostgreSQL: %s  |  SQLite: ?
    """
    if USE_POSTGRES:
        # PostgreSQL sudah pakai %s — tidak perlu ganti
        pass
    else:
        # SQLite pakai ? — ganti %s → ?
        sql = sql.replace("%s", "?")

    conn = _get_conn()
    try:
        if USE_POSTGRES:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cur = conn.cursor()

        cur.execute(sql, params or ())

        if fetch:
            rows = cur.fetchall()
            conn.commit() if USE_POSTGRES else conn.commit()
            return [dict(r) for r in rows]
        else:
            conn.commit()
            return cur.rowcount
    except Exception as e:
        conn.rollback() if USE_POSTGRES else None
        print(f"[DB ERROR] {e}\nSQL: {sql}\nParams: {params}")
        return None
    finally:
        if USE_POSTGRES:
            conn.close()

# ==================================================
# INIT — buat semua tabel jika belum ada
# ==================================================
SCHEMA_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id           SERIAL PRIMARY KEY,
    ts           BIGINT      NOT NULL,
    pair         VARCHAR(20) NOT NULL,
    timeframe    VARCHAR(10) NOT NULL,
    action       VARCHAR(10) NOT NULL,
    score        INTEGER     NOT NULL,
    confluence   INTEGER,
    confidence   VARCHAR(30),
    entry        REAL        NOT NULL,
    tp1          REAL,
    tp2          REAL,
    sl1          REAL,
    sl2          REAL,
    rr1          REAL,
    rr2          REAL,
    atr          REAL,
    rsi          REAL,
    macd_hist    REAL,
    stoch_k      REAL,
    bb_pos       REAL,
    vol_ratio    REAL,
    rsi_div      VARCHAR(20),
    lstm         VARCHAR(20),
    session      VARCHAR(20),
    mtf_bias     VARCHAR(20),
    of_bias      VARCHAR(20),
    in_fvg       VARCHAR(20),
    cats_json    TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_SIGNALS_SQLITE = """
CREATE TABLE IF NOT EXISTS signals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER     NOT NULL,
    pair         TEXT        NOT NULL,
    timeframe    TEXT        NOT NULL,
    action       TEXT        NOT NULL,
    score        INTEGER     NOT NULL,
    confluence   INTEGER,
    confidence   TEXT,
    entry        REAL        NOT NULL,
    tp1          REAL,
    tp2          REAL,
    sl1          REAL,
    sl2          REAL,
    rr1          REAL,
    rr2          REAL,
    atr          REAL,
    rsi          REAL,
    macd_hist    REAL,
    stoch_k      REAL,
    bb_pos       REAL,
    vol_ratio    REAL,
    rsi_div      TEXT,
    lstm         TEXT,
    session      TEXT,
    mtf_bias     TEXT,
    of_bias      TEXT,
    in_fvg       TEXT,
    cats_json    TEXT,
    created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_DIV_ALERTS = """
CREATE TABLE IF NOT EXISTS div_alerts (
    id           {pk},
    ts           {int_t}     NOT NULL,
    pair         {str_t}     NOT NULL,
    timeframe    {str_t}     NOT NULL,
    div_type     {str_t}     NOT NULL,
    strength     REAL,
    score        INTEGER,
    rsi_now      REAL,
    rsi_prev     REAL,
    price_now    REAL,
    price_prev   REAL,
    bos_fresh    {bool_t},
    candle_ok    {bool_t},
    created_at   {ts_t} DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_SCORE_HISTORY = """
CREATE TABLE IF NOT EXISTS score_history (
    id           {pk},
    ts           {int_t}     NOT NULL,
    pair         {str_t}     NOT NULL,
    timeframe    {str_t}     NOT NULL,
    action       {str_t}     NOT NULL,
    score        INTEGER,
    cat_trend    INTEGER,
    cat_momentum INTEGER,
    cat_smc      INTEGER,
    cat_orderflow INTEGER,
    cat_fvg      INTEGER,
    cat_mtf      INTEGER,
    cat_session  INTEGER,
    cat_bb       INTEGER,
    adx          REAL,
    rsi          REAL,
    vol_ratio    REAL,
    session      {str_t},
    created_at   {ts_t} DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_OUTCOMES = """
CREATE TABLE IF NOT EXISTS outcomes (
    id           {pk},
    signal_id    INTEGER,
    pair         {str_t},
    timeframe    {str_t},
    action       {str_t},
    entry        REAL,
    tp1          REAL,
    tp2          REAL,
    sl1          REAL,
    sl2          REAL,
    outcome      {str_t},
    pips_result  REAL,
    rr_achieved  REAL,
    bars_held    INTEGER,
    created_at   {ts_t} DEFAULT CURRENT_TIMESTAMP
);
"""

# outcome.outcome values: 'TP1', 'TP2', 'SL1', 'SL2', 'OPEN', 'MANUAL_CLOSE'

def _fmt_schema(template: str) -> str:
    """Format schema template sesuai backend."""
    if USE_POSTGRES:
        return template.format(
            pk     = "SERIAL PRIMARY KEY",
            int_t  = "BIGINT",
            str_t  = "VARCHAR(30)",
            bool_t = "BOOLEAN",
            ts_t   = "TIMESTAMP",
        )
    else:
        return template.format(
            pk     = "INTEGER PRIMARY KEY AUTOINCREMENT",
            int_t  = "INTEGER",
            str_t  = "TEXT",
            bool_t = "INTEGER",
            ts_t   = "TEXT",
        )

def init_db():
    """Buat semua tabel. Panggil sekali saat startup."""
    schemas = [
        SCHEMA_SIGNALS if USE_POSTGRES else SCHEMA_SIGNALS_SQLITE,
        _fmt_schema(SCHEMA_DIV_ALERTS),
        _fmt_schema(SCHEMA_SCORE_HISTORY),
        _fmt_schema(SCHEMA_OUTCOMES),
    ]
    for sql in schemas:
        _execute(sql)
    print(f"✅ DB tables initialized ({'PostgreSQL' if USE_POSTGRES else 'SQLite'})")

# ==================================================
# LOG SIGNAL
# ==================================================
def log_signal(data: dict) -> Optional[int]:
    """
    Simpan signal ke tabel signals.
    Return inserted row ID (untuk link ke outcomes nanti).
    """
    cats = data.get("cats", {})
    fvg  = data.get("fvg",  {})

    in_fvg = (
        "BULL" if fvg.get("in_bull_fvg") else
        "BEAR" if fvg.get("in_bear_fvg") else
        "OUT"
    )

    sql = """
        INSERT INTO signals
            (ts, pair, timeframe, action, score, confluence, confidence,
             entry, tp1, tp2, sl1, sl2, rr1, rr2, atr, rsi, macd_hist,
             stoch_k, bb_pos, vol_ratio, rsi_div, lstm, session,
             mtf_bias, of_bias, in_fvg, cats_json)
        VALUES
            (%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s)
        RETURNING id
    """
    # SQLite tidak support RETURNING → pakai lastrowid
    if not USE_POSTGRES:
        sql = sql.replace(" RETURNING id", "")

    params = (
        int(time.time()),
        data["pair"], data["timeframe"], data["action"],
        data["score"], data.get("confluence"), data.get("confidence"),
        data["entry"],
        data.get("tp1"), data.get("tp2"),
        data.get("sl1"), data.get("sl2"),
        data.get("rr1"), data.get("rr2"),
        data.get("atr"), data.get("rsi"), data.get("macd"),
        data.get("stoch_k"), data.get("bb_pos"), data.get("vol_ratio"),
        data.get("rsi_div"),
        data.get("lstm"),
        data.get("session", {}).get("session"),
        data.get("mtf",      {}).get("bias"),
        data.get("orderflow",{}).get("bias"),
        in_fvg,
        json.dumps(cats),
    )

    try:
        if USE_POSTGRES:
            conn = _get_conn()
            cur  = conn.cursor()
            cur.execute(sql.replace("%s", "%s"), params)
            row_id = cur.fetchone()[0]
            conn.commit()
            conn.close()
            return row_id
        else:
            conn = _get_conn()
            sql_lite = sql.replace("%s", "?")
            cur  = conn.execute(sql_lite, params)
            conn.commit()
            return cur.lastrowid
    except Exception as e:
        print(f"[DB log_signal ERROR] {e}")
        return None

# ==================================================
# LOG SCORE HISTORY  (untuk ML training data)
# ==================================================
def log_score_history(data: dict):
    cats = data.get("cats", {})
    tr   = data.get("trend", {})

    sql = """
        INSERT INTO score_history
            (ts, pair, timeframe, action, score,
             cat_trend, cat_momentum, cat_smc, cat_orderflow,
             cat_fvg, cat_mtf, cat_session, cat_bb,
             adx, rsi, vol_ratio, session)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (
        int(time.time()),
        data["pair"], data["timeframe"], data["action"], data["score"],
        cats.get("TREND"), cats.get("MOMENTUM"), cats.get("SMC"),
        cats.get("ORDERFLOW"), cats.get("FVG"), cats.get("MTF"),
        cats.get("SESSION"), cats.get("BB"),
        tr.get("adx"),
        data.get("rsi"),
        data.get("vol_ratio"),
        data.get("session", {}).get("session"),
    )
    _execute(sql, params)

# ==================================================
# LOG DIV ALERT
# ==================================================
def log_div_alert(pair: str, timeframe: str, div_det: dict, signal_data: dict):
    bos    = div_det.get("bos", {})
    candle = div_det.get("candle", {})

    sql = """
        INSERT INTO div_alerts
            (ts, pair, timeframe, div_type, strength, score,
             rsi_now, rsi_prev, price_now, price_prev,
             bos_fresh, candle_ok)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    is_bull  = div_det["type"] == "BULLISH_DIV"
    params = (
        int(time.time()),
        pair, timeframe, div_det["type"],
        div_det.get("strength"), signal_data.get("score"),
        div_det.get("rsi_now"), div_det.get("rsi_prev"),
        div_det.get("price_now"), div_det.get("price_prev"),
        1 if (bos.get("fresh_bull") if is_bull else bos.get("fresh_bear")) else 0,
        1 if candle.get("confirmed") else 0,
    )
    _execute(sql, params)

# ==================================================
# LOG OUTCOME  (call dari webhook / manual command)
# ==================================================
def log_outcome(
    signal_id: int,
    pair: str,
    timeframe: str,
    action: str,
    entry: float,
    tp1: float,
    tp2: float,
    sl1: float,
    sl2: float,
    outcome: str,        # 'TP1' | 'TP2' | 'SL1' | 'SL2' | 'MANUAL_CLOSE'
    pips_result: float,
    rr_achieved: float,
    bars_held: int = 0,
):
    sql = """
        INSERT INTO outcomes
            (signal_id, pair, timeframe, action, entry,
             tp1, tp2, sl1, sl2, outcome, pips_result, rr_achieved, bars_held)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    _execute(sql, (
        signal_id, pair, timeframe, action, entry,
        tp1, tp2, sl1, sl2, outcome, pips_result, rr_achieved, bars_held,
    ))


# ==================================================
# QUERY HELPERS  (untuk analytics & ML)
# ==================================================
def get_recent_signals(pair: str = None, limit: int = 100) -> list:
    """Ambil signal terbaru. Filter pair opsional."""
    if pair:
        sql = "SELECT * FROM signals WHERE pair=%s ORDER BY ts DESC LIMIT %s"
        return _execute(sql, (pair, limit), fetch=True) or []
    else:
        sql = "SELECT * FROM signals ORDER BY ts DESC LIMIT %s"
        return _execute(sql, (limit,), fetch=True) or []

def get_win_rate(pair: str = None, timeframe: str = None) -> dict:
    """
    Hitung win rate dari tabel outcomes.
    Win = TP1 atau TP2 hit.
    """
    conditions = []
    params     = []
    if pair:
        conditions.append("pair=%s"); params.append(pair)
    if timeframe:
        conditions.append("timeframe=%s"); params.append(timeframe)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    sql_total = f"SELECT COUNT(*) as n FROM outcomes {where}"
    sql_wins  = f"SELECT COUNT(*) as n FROM outcomes {where} {'AND' if where else 'WHERE'} outcome IN ('TP1','TP2')"

    total_rows = _execute(sql_total, params or None, fetch=True)
    win_rows   = _execute(sql_wins,  params or None, fetch=True)

    total = (total_rows[0]["n"] if total_rows else 0) or 0
    wins  = (win_rows[0]["n"]   if win_rows   else 0) or 0

    return {
        "total":    total,
        "wins":     wins,
        "losses":   total - wins,
        "win_rate": round(wins / total * 100, 2) if total else 0.0,
    }

def get_training_data(min_rows: int = 200) -> list:
    """
    Ambil data gabungan signals + outcomes untuk ML training.
    Hanya baris yang sudah ada outcome-nya.
    """
    sql = """
        SELECT
            s.score, s.cat_trend, s.cat_momentum, s.cat_smc,
            s.cat_orderflow, s.cat_fvg, s.cat_mtf, s.cat_session,
            s.adx, s.rsi, s.vol_ratio, s.action,
            o.outcome, o.rr_achieved, o.bars_held
        FROM score_history s
        JOIN outcomes o ON s.pair = o.pair
            AND s.timeframe = o.timeframe
            AND ABS(s.ts - o.created_at) < 3600
        ORDER BY s.ts DESC
        LIMIT %s
    """
    rows = _execute(sql, (min_rows * 2,), fetch=True) or []
    return rows

def get_db_stats() -> dict:
    """Ringkasan isi database."""
    counts = {}
    for table in ("signals", "div_alerts", "score_history", "outcomes"):
        rows = _execute(f"SELECT COUNT(*) as n FROM {table}", fetch=True)
        counts[table] = (rows[0]["n"] if rows else 0) or 0
    return counts
