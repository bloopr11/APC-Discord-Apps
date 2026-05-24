import os
import time
import json
import sqlite3
import threading
from typing import Optional

DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL and DATABASE_URL.startswith("postgresql"))

_pg_pool = None
_pg_pool_lock = threading.Lock()

def _get_pg_pool():
    global _pg_pool
    if not USE_POSTGRES:
        return None
    with _pg_pool_lock:
        if _pg_pool is None:
            try:
                from psycopg2 import pool
                _pg_pool = pool.SimpleConnectionPool(1, 10, DATABASE_URL, connect_timeout=10)
                print("✅ PostgreSQL connection pool created")
            except Exception as e:
                print(f"❌ PG pool error: {e}")
                return None
    return _pg_pool

if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        _get_pg_pool()
        print("✅ DB: PostgreSQL mode with pool")
    except ImportError:
        USE_POSTGRES = False
        print("⚠️ psycopg2 missing -> fallback SQLite")
else:
    print("ℹ️ SQLite mode")

SQLITE_PATH = os.getenv("SQLITE_PATH", "trading_bot.db")
_local = threading.local()

def _get_conn():
    if USE_POSTGRES:
        pool = _get_pg_pool()
        if pool:
            return pool.getconn()
        else:
            return psycopg2.connect(DATABASE_URL)
    else:
        if not hasattr(_local, "conn") or _local.conn is None:
            _local.conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            _local.conn.row_factory = sqlite3.Row
        return _local.conn

def _return_conn(conn):
    if USE_POSTGRES:
        pool = _get_pg_pool()
        if pool:
            pool.putconn(conn)
        else:
            conn.close()

def _execute(sql, params=None, fetch=False):
    if USE_POSTGRES:
        pass
    else:
        sql = sql.replace("%s", "?")
    conn = None
    try:
        conn = _get_conn()
        if USE_POSTGRES:
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cur = conn.cursor()
        cur.execute(sql, params or ())
        if fetch:
            rows = cur.fetchall()
            conn.commit()
            return [dict(r) for r in rows]
        else:
            conn.commit()
            return cur.rowcount
    except Exception as e:
        if conn:
            conn.rollback()
        print(f"[DB ERROR] {e}")
        return None
    finally:
        if cur:
            cur.close()
        if conn:
            _return_conn(conn)

def init_db():
    if USE_POSTGRES:
        schemas = ["""
            CREATE TABLE IF NOT EXISTS signals (
                id SERIAL PRIMARY KEY, ts BIGINT NOT NULL, pair VARCHAR(20) NOT NULL,
                timeframe VARCHAR(10) NOT NULL, action VARCHAR(10) NOT NULL, score INTEGER NOT NULL,
                confluence INTEGER, confidence VARCHAR(30), entry REAL NOT NULL,
                tp1 REAL, tp2 REAL, sl1 REAL, sl2 REAL, rr1 REAL, rr2 REAL,
                atr REAL, rsi REAL, macd_hist REAL, stoch_k REAL, bb_pos REAL, vol_ratio REAL,
                rsi_div VARCHAR(20), lstm VARCHAR(20), session VARCHAR(20), mtf_bias VARCHAR(20),
                of_bias VARCHAR(20), in_fvg VARCHAR(20), cats_json TEXT, created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS div_alerts (
                id SERIAL PRIMARY KEY, ts BIGINT NOT NULL, pair VARCHAR(20), timeframe VARCHAR(10),
                div_type VARCHAR(20), strength REAL, score INTEGER, rsi_now REAL, rsi_prev REAL,
                price_now REAL, price_prev REAL, bos_fresh BOOLEAN, candle_ok BOOLEAN, created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS score_history (
                id SERIAL PRIMARY KEY, ts BIGINT NOT NULL, pair VARCHAR(20), timeframe VARCHAR(10),
                action VARCHAR(10), score INTEGER, cat_trend INTEGER, cat_momentum INTEGER,
                cat_smc INTEGER, cat_orderflow INTEGER, cat_fvg INTEGER, cat_mtf INTEGER,
                cat_session INTEGER, cat_bb INTEGER, adx REAL, rsi REAL, vol_ratio REAL,
                session VARCHAR(20), created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                id SERIAL PRIMARY KEY, signal_id INTEGER, pair VARCHAR(20), timeframe VARCHAR(10),
                action VARCHAR(10), entry REAL, tp1 REAL, tp2 REAL, sl1 REAL, sl2 REAL,
                outcome VARCHAR(20), pips_result REAL, rr_achieved REAL, bars_held INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """]
        for sql in schemas:
            _execute(sql)
    else:
        # SQLite schemas (sederhana, sama seperti kode asli)
        _execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, pair TEXT, timeframe TEXT,
                action TEXT, score INTEGER, confluence INTEGER, confidence TEXT, entry REAL,
                tp1 REAL, tp2 REAL, sl1 REAL, sl2 REAL, rr1 REAL, rr2 REAL,
                atr REAL, rsi REAL, macd_hist REAL, stoch_k REAL, bb_pos REAL, vol_ratio REAL,
                rsi_div TEXT, lstm TEXT, session TEXT, mtf_bias TEXT, of_bias TEXT, in_fvg TEXT,
                cats_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS div_alerts (...);
            CREATE TABLE IF NOT EXISTS score_history (...);
            CREATE TABLE IF NOT EXISTS outcomes (...);
        """)
    print(f"✅ DB initialized ({'PostgreSQL' if USE_POSTGRES else 'SQLite'})")

def log_signal(data: dict) -> Optional[int]:
    cats = data.get("cats", {})
    fvg = data.get("fvg", {})
    in_fvg = "BULL" if fvg.get("in_bull_fvg") else "BEAR" if fvg.get("in_bear_fvg") else "OUT"
    sql = """
        INSERT INTO signals
        (ts, pair, timeframe, action, score, confluence, confidence,
         entry, tp1, tp2, sl1, sl2, rr1, rr2, atr, rsi, macd_hist,
         stoch_k, bb_pos, vol_ratio, rsi_div, lstm, session,
         mtf_bias, of_bias, in_fvg, cats_json)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (int(time.time()), data["pair"], data["timeframe"], data["action"],
              data["score"], data.get("confluence"), data.get("confidence"),
              data["entry"], data.get("tp1"), data.get("tp2"), data.get("sl1"), data.get("sl2"),
              data.get("rr1"), data.get("rr2"), data.get("atr"), data.get("rsi"), data.get("macd"),
              data.get("stoch_k"), data.get("bb_pos"), data.get("vol_ratio"), data.get("rsi_div"),
              data.get("lstm"), data.get("session", {}).get("session"),
              data.get("mtf", {}).get("bias"), data.get("orderflow", {}).get("bias"),
              in_fvg, json.dumps(cats))
    if USE_POSTGRES:
        res = _execute(sql + " RETURNING id", params, fetch=True)
        return res[0]["id"] if res else None
    else:
        _execute(sql, params)
        conn = _get_conn()
        last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        _return_conn(conn)
        return last_id

def log_score_history(data: dict):
    cats = data.get("cats", {})
    tr = data.get("trend", {})
    sql = """
        INSERT INTO score_history
        (ts, pair, timeframe, action, score, cat_trend, cat_momentum, cat_smc,
         cat_orderflow, cat_fvg, cat_mtf, cat_session, cat_bb,
         adx, rsi, vol_ratio, session)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (int(time.time()), data["pair"], data["timeframe"], data["action"], data["score"],
              cats.get("TREND"), cats.get("MOMENTUM"), cats.get("SMC"),
              cats.get("ORDERFLOW"), cats.get("FVG"), cats.get("MTF"),
              cats.get("SESSION"), cats.get("BB"),
              tr.get("adx"), data.get("rsi"), data.get("vol_ratio"),
              data.get("session", {}).get("session"))
    _execute(sql, params)

def log_div_alert(pair, timeframe, div_det, signal_data):
    bos = div_det.get("bos", {})
    candle = div_det.get("candle", {})
    is_bull = div_det["type"] == "BULLISH_DIV"
    sql = """
        INSERT INTO div_alerts
        (ts, pair, timeframe, div_type, strength, score,
         rsi_now, rsi_prev, price_now, price_prev,
         bos_fresh, candle_ok)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    params = (int(time.time()), pair, timeframe, div_det["type"], div_det.get("strength"),
              signal_data.get("score"), div_det.get("rsi_now"), div_det.get("rsi_prev"),
              div_det.get("price_now"), div_det.get("price_prev"),
              1 if (bos.get("fresh_bull") if is_bull else bos.get("fresh_bear")) else 0,
              1 if candle.get("confirmed") else 0)
    _execute(sql, params)

def log_outcome(signal_id, pair, timeframe, action, entry, tp1, tp2, sl1, sl2, outcome, pips_result, rr_achieved, bars_held=0):
    sql = """
        INSERT INTO outcomes
        (signal_id, pair, timeframe, action, entry, tp1, tp2, sl1, sl2, outcome, pips_result, rr_achieved, bars_held)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """
    _execute(sql, (signal_id, pair, timeframe, action, entry, tp1, tp2, sl1, sl2, outcome, pips_result, rr_achieved, bars_held))

def get_recent_signals(pair=None, limit=100):
    if pair:
        sql = "SELECT * FROM signals WHERE pair=%s ORDER BY ts DESC LIMIT %s"
        return _execute(sql, (pair, limit), fetch=True) or []
    else:
        sql = "SELECT * FROM signals ORDER BY ts DESC LIMIT %s"
        return _execute(sql, (limit,), fetch=True) or []

def get_win_rate(pair=None, timeframe=None):
    conditions = []
    params = []
    if pair:
        conditions.append("pair=%s"); params.append(pair)
    if timeframe:
        conditions.append("timeframe=%s"); params.append(timeframe)
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    sql_total = f"SELECT COUNT(*) as n FROM outcomes {where}"
    sql_wins = f"SELECT COUNT(*) as n FROM outcomes {where} {'AND' if where else 'WHERE'} outcome IN ('TP1','TP2')"
    total_rows = _execute(sql_total, params or None, fetch=True)
    win_rows = _execute(sql_wins, params or None, fetch=True)
    total = total_rows[0]["n"] if total_rows else 0
    wins = win_rows[0]["n"] if win_rows else 0
    return {"total": total, "wins": wins, "losses": total - wins, "win_rate": round(wins/total*100,2) if total else 0.0}

def get_training_data(min_rows=200):
    if USE_POSTGRES:
        sql = """
            SELECT s.score, s.cat_trend, s.cat_momentum, s.cat_smc,
                   s.cat_orderflow, s.cat_fvg, s.cat_mtf, s.cat_session,
                   s.adx, s.rsi, s.vol_ratio, s.action,
                   o.outcome, o.rr_achieved, o.bars_held
            FROM score_history s
            JOIN outcomes o ON s.pair = o.pair AND s.timeframe = o.timeframe
                AND ABS(s.ts - EXTRACT(EPOCH FROM o.created_at)::BIGINT) < 3600
            ORDER BY s.ts DESC LIMIT %s
        """
    else:
        sql = """
            SELECT s.score, s.cat_trend, s.cat_momentum, s.cat_smc,
                   s.cat_orderflow, s.cat_fvg, s.cat_mtf, s.cat_session,
                   s.adx, s.rsi, s.vol_ratio, s.action,
                   o.outcome, o.rr_achieved, o.bars_held
            FROM score_history s
            JOIN outcomes o ON s.pair = o.pair AND s.timeframe = o.timeframe
                AND ABS(s.ts - CAST(strftime('%%s', o.created_at) AS INTEGER)) < 3600
            ORDER BY s.ts DESC LIMIT %s
        """
    rows = _execute(sql, (min_rows*2,), fetch=True) or []
    return rows

def get_db_stats():
    counts = {}
    for table in ["signals", "div_alerts", "score_history", "outcomes"]:
        try:
            res = _execute(f"SELECT COUNT(*) as n FROM {table}", fetch=True)
            counts[table] = res[0]["n"] if res else 0
        except:
            counts[table] = 0
    return counts

def close_db_pool():
    global _pg_pool
    if USE_POSTGRES and _pg_pool:
        _pg_pool.closeall()
        print("✅ PostgreSQL pool closed")
