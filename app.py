import discord
from discord import app_commands
from discord.ui import View, Button, Select
import asyncio
import os
import functools
import time
import traceback
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

# ==================================================
# NEW MODULE IMPORTS  (db_logger, regime, ml_scorer)
# ==================================================
from db_logger  import (
    init_db, log_signal, log_score_history, log_div_alert,
    log_outcome, get_recent_signals, get_win_rate, get_db_stats,
    get_training_data,
)
from regime     import detect_regime, regime_score_modifier, should_skip, regime_embed_value
from smc_engine import smc_full, smc_score_modifier, smc_embed_value
from ml_scorer  import (
    adaptive_score, maybe_retrain, ml_status, ml_embed_value,
    train, load_model, _model_meta, ML_MIN_ROWS,
)

# ==================================================
# ENV
# ==================================================
TOKEN      = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))
TV_USERNAME = os.getenv("TV_USERNAME", "")
TV_PASSWORD = os.getenv("TV_PASSWORD", "")

# ==================================================
# PAIRS config
# ==================================================
from pairs_config import PAIRS, TIMEFRAMES, MTF_FETCH, ASIA_ACTIVE, DEFAULT_TF

AUTO_SIGNAL    = False
AUTO_MIN_SCORE = 75

# ==================================================
# TRADINGVIEW WEBSOCKET
# ==================================================
import json, re, random, string

def _tv_token() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))

def _tv_msg(func: str, args: list) -> str:
    payload = json.dumps({"m": func, "p": args}, separators=(",", ":"))
    return f"~m~{len(payload)}~m~{payload}"

def _tv_parse(raw: str) -> list:
    msgs = []
    for chunk in re.findall(r"~m~\d+~m~(.+?)(?=~m~\d+~m~|$)", raw, re.DOTALL):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("~h~"):
            continue
        try:
            msgs.append(json.loads(chunk))
        except Exception:
            pass
    return msgs

def _fetch_tv_ws(exchange: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    import websockets.sync.client as wsc

    full_sym  = f"{exchange}:{symbol}"
    cs_token  = f"cs_{_tv_token()}"
    qs_token  = f"qs_{_tv_token()}"

    WS_URL  = "wss://data.tradingview.com/socket.io/websocket"
    HEADERS = {
        "Origin":     "https://www.tradingview.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
    }

    candles = []

    try:
        with wsc.connect(WS_URL, additional_headers=HEADERS,
                         open_timeout=12, close_timeout=5) as ws:
            ws.send(_tv_msg("set_auth_token",    ["unauthorized_user_token"]))
            ws.send(_tv_msg("chart_create_session", [cs_token, ""]))
            ws.send(_tv_msg("quote_create_session", [qs_token]))
            ws.send(_tv_msg("resolve_symbol", [cs_token, "sds_sym_1",
                f'={{"symbol":"{full_sym}","adjustment":"splits"}}']))
            ws.send(_tv_msg("create_series", [
                cs_token, "sds_1", "s1", "sds_sym_1", interval, bars, ""
            ]))

            deadline  = time.time() + 20
            completed = False
            while time.time() < deadline and not completed:
                try:
                    raw = ws.recv(timeout=6)
                except Exception:
                    break

                if "~h~" in raw:
                    ws.send(f"~m~{len(raw)}~m~{raw}")
                    continue

                for msg in _tv_parse(raw):
                    m = msg.get("m", "")
                    if m == "timescale_update":
                        bars_data = (
                            (msg.get("p") or [{}])[1] or {}
                        ).get("sds_1", {}).get("s", [])
                        for b in bars_data:
                            v = b.get("v", [])
                            if len(v) >= 5:
                                candles.append({
                                    "ts":     v[0],
                                    "Open":   v[1],
                                    "High":   v[2],
                                    "Low":    v[3],
                                    "Close":  v[4],
                                    "Volume": v[5] if len(v) > 5 else 0,
                                })
                    elif m == "series_completed":
                        completed = True
                        break
                    elif m == "series_error":
                        raise ValueError(f"TV series_error: {msg}")

    except Exception as e:
        err = str(e)
        if "429" in err:
            raise ConnectionError("TV_RATE_LIMITED")
        raise

    if not candles:
        raise ValueError("No candles received from TradingView WS")

    df = pd.DataFrame(candles)
    df["Date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("Date").drop(columns=["ts"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

# ==================================================
# YAHOO FINANCE FALLBACK
# ==================================================
def _fetch_yf(yf_symbol: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(yf_symbol, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

# ==================================================
# TV AVAILABLE CHECK
# ==================================================
_tv_ok: bool | None = None

def _check_tv_available() -> bool:
    global _tv_ok
    if _tv_ok is not None:
        return _tv_ok
    try:
        import websockets.sync.client  # noqa
        _tv_ok = True
        print("✅ TradingView WS: websockets tersedia")
    except ImportError:
        _tv_ok = False
        print("⚠️  websockets tidak tersedia → Yahoo Finance only")
    return _tv_ok

# ==================================================
# TV RATE LIMIT
# ==================================================
import threading
from concurrent.futures import ThreadPoolExecutor

_tv_lock       = threading.Lock()
_tv_last_call  = 0.0
TV_MIN_INTERVAL = 3.0

_tv_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tv_fetch")

def _tv_throttle_sync():
    global _tv_last_call
    with _tv_lock:
        elapsed = time.time() - _tv_last_call
        if elapsed < TV_MIN_INTERVAL:
            time.sleep(TV_MIN_INTERVAL - elapsed)
        _tv_last_call = time.time()

async def run_tv_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _tv_executor, functools.partial(func, *args, **kwargs)
    )

def _get_tv_semaphore():
    return asyncio.Semaphore(999)

# ==================================================
# IN-MEMORY CACHE
# ==================================================
_data_cache: dict = {}
CACHE_TTL     = 180
CACHE_TTL_HTF = 1800

def _cache_get(cache_key: str, ttl: int):
    now = time.time()
    if cache_key in _data_cache:
        ts, df = _data_cache[cache_key]
        if now - ts < ttl:
            return df
    return None

def _cache_set(cache_key: str, df):
    _data_cache[cache_key] = (time.time(), df)

# ==================================================
# PUBLIC DATA API
# ==================================================
def get_data(pair_key: str, timeframe: str = DEFAULT_TF) -> pd.DataFrame:
    cfg       = TIMEFRAMES.get(timeframe, TIMEFRAMES[DEFAULT_TF])
    pair_info = PAIRS[pair_key]
    cache_key = f"{pair_key}_{timeframe}"

    cached = _cache_get(cache_key, CACHE_TTL)
    if cached is not None:
        return cached

    df = None
    if _check_tv_available():
        try:
            _tv_throttle_sync()
            df = _fetch_tv_ws(
                pair_info["tv"][0], pair_info["tv"][1],
                cfg["tv_interval"], cfg["bars"],
            )
            print(f"[TV✅] {pair_key} {timeframe} {len(df)} bars")
        except ConnectionError:
            print(f"[TV⚠️ ] {pair_key} {timeframe}: rate limited → Yahoo")
        except Exception as e:
            print(f"[TV❌] {pair_key} {timeframe}: {e} → Yahoo")

    if df is None or len(df) < 30:
        df = _fetch_yf(pair_info["yf"], cfg["yf_period"], cfg["yf_interval"])

    _cache_set(cache_key, df)
    return df

def get_htf_data(pair_key: str, tf: str) -> pd.DataFrame:
    cfg       = MTF_FETCH.get(tf, MTF_FETCH["1h"])
    pair_info = PAIRS[pair_key]
    cache_key = f"{pair_key}_{tf}_htf"

    cached = _cache_get(cache_key, CACHE_TTL_HTF)
    if cached is not None:
        return cached

    df = None
    if _check_tv_available():
        try:
            _tv_throttle_sync()
            df = _fetch_tv_ws(
                pair_info["tv"][0], pair_info["tv"][1],
                cfg["tv_interval"], cfg["bars"],
            )
        except ConnectionError:
            print(f"[TV HTF⚠️ ] {pair_key} {tf}: rate limited → Yahoo")
        except Exception as e:
            print(f"[TV HTF❌] {pair_key} {tf}: {e} → Yahoo")

    if df is None or len(df) < 30:
        df = _fetch_yf(pair_info["yf"], cfg["yf_period"], cfg["yf_interval"])

    _cache_set(cache_key, df)
    return df

def _fetch_with_throttle(exchange: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    _tv_throttle_sync()
    return _fetch_tv_ws(exchange, symbol, interval, bars)

# ==================================================
# DISCORD CLIENT
# ==================================================
class Bot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

client = Bot()

# ==================================================
# ASYNC HELPER
# ==================================================
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, *args, **kwargs)
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series):
    ema12       = ema(series, 12)
    ema26       = ema(series, 26)
    macd_line   = ema12 - ema26
    signal_line = ema(macd_line, 9)
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min  = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    stoch_k  = 100 * (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    stoch_d  = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    return sma + std_dev * std, sma, sma - std_dev * std

def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return 1.0
    avg_vol = df["Volume"].rolling(period).mean().iloc[-1]
    cur_vol = df["Volume"].iloc[-1]
    return float(cur_vol / avg_vol) if avg_vol else 1.0

# ==================================================
# ① DELTA VOLUME PROXY
# ==================================================
def delta_orderflow(df: pd.DataFrame, period: int = 20) -> dict:
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return {
            "delta":        0.0,
            "cvd_slope":    0.0,
            "bias":         "NEUTRAL",
            "buy_vol_pct":  50.0,
            "absorption":   False,
        }

    close  = df["Close"]
    open_  = df["Open"]
    vol    = df["Volume"]

    direction     = np.where(close >= open_, 1.0, -1.0)
    signed_vol    = vol * direction
    cvd           = signed_vol.cumsum()

    cvd_tail      = cvd.iloc[-period:]
    x             = np.arange(len(cvd_tail))
    if len(x) > 1:
        slope = float(np.polyfit(x, cvd_tail.values, 1)[0])
    else:
        slope = 0.0

    last_delta    = float(signed_vol.iloc[-1])

    buy_vol       = vol[direction == 1.0].iloc[-period:].sum()
    total_vol     = vol.iloc[-period:].sum()
    buy_pct       = float(buy_vol / total_vol * 100) if total_vol > 0 else 50.0

    price_range   = float((close - open_).abs().iloc[-3:].mean())
    vol_mean      = float(vol.rolling(20).mean().iloc[-1]) or 1
    absorption    = (float(vol.iloc[-1]) > 1.5 * vol_mean) and (price_range < float(atr(df).iloc[-1]) * 0.3)

    bias = "BUY" if slope > 0 else "SELL" if slope < 0 else "NEUTRAL"

    return {
        "delta":       last_delta,
        "cvd_slope":   slope,
        "bias":        bias,
        "buy_vol_pct": buy_pct,
        "absorption":  absorption,
        "cvd":         cvd,
    }

# ==================================================
# ② FVG ENGINE
# ==================================================
def fvg_engine(df: pd.DataFrame, lookback: int = 50) -> dict:
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    n      = len(df)

    bull_fvgs = []
    bear_fvgs = []

    start = max(2, n - lookback)
    for i in range(start, n - 1):
        if lows[i] > highs[i - 2]:
            top    = lows[i]
            bottom = highs[i - 2]
            if closes[-1] > bottom:
                bull_fvgs.append({"top": top, "bottom": bottom, "idx": i})

        if highs[i] < lows[i - 2]:
            top    = lows[i - 2]
            bottom = highs[i]
            if closes[-1] < top:
                bear_fvgs.append({"top": top, "bottom": bottom, "idx": i})

    price = closes[-1]

    in_bull_fvg = any(f["bottom"] <= price <= f["top"] for f in bull_fvgs)
    in_bear_fvg = any(f["bottom"] <= price <= f["top"] for f in bear_fvgs)

    nearest_bull = bull_fvgs[-1] if bull_fvgs else None
    nearest_bear = bear_fvgs[-1] if bear_fvgs else None

    return {
        "bull_fvgs":    bull_fvgs,
        "bear_fvgs":    bear_fvgs,
        "in_bull_fvg":  in_bull_fvg,
        "in_bear_fvg":  in_bear_fvg,
        "nearest_bull": nearest_bull,
        "nearest_bear": nearest_bear,
        "count_bull":   len(bull_fvgs),
        "count_bear":   len(bear_fvgs),
    }

# ==================================================
# ③ MTF BIAS
# ==================================================
MTF_MAP = {
    "5m":  ["1h",  "4h"],
    "15m": ["1h",  "4h"],
    "1h":  ["4h",  "1d"],
    "4h":  ["1d",  "1wk"],
}

def mtf_bias(pair_key: str, base_tf: str) -> dict:
    htf_list = MTF_MAP.get(base_tf, ["1h", "4h"])
    results  = {}
    total_score = 0
    weights     = [0.6, 0.4]

    for tf, w in zip(htf_list, weights):
        try:
            df_h   = get_htf_data(pair_key, tf)
            close  = df_h["Close"]
            price  = float(close.iloc[-1])

            e8     = float(ema(close, 8).iloc[-1])
            e21    = float(ema(close, 21).iloc[-1])
            e50    = float(ema(close, 50).iloc[-1])
            e200   = float(ema(close, 200).iloc[-1])
            rsi_v  = float(rsi(close).iloc[-1])
            _, _, mh = macd(close)
            macd_v = float(mh.iloc[-1])

            s = 0
            if price > e8 > e21 > e50:   s += 40
            elif price < e8 < e21 < e50: s -= 40
            elif price > e21:            s += 20
            elif price < e21:            s -= 20

            if price > e200:  s += 20
            else:             s -= 20

            if rsi_v > 55:   s += 20
            elif rsi_v < 45: s -= 20

            if macd_v > 0:   s += 20
            else:             s -= 20

            bias_lbl = "BULLISH" if s > 20 else "BEARISH" if s < -20 else "NEUTRAL"
            results[tf] = {
                "score": s, "bias": bias_lbl,
                "rsi": rsi_v, "price": price,
                "e50": e50, "e200": e200,
            }
            total_score += s * w

        except Exception as ex:
            results[tf] = {"score": 0, "bias": "NEUTRAL", "error": str(ex)}

    combined_bias = "BULLISH" if total_score > 15 else "BEARISH" if total_score < -15 else "NEUTRAL"
    return {
        "score":    total_score,
        "bias":     combined_bias,
        "detail":   results,
        "htf_list": htf_list,
    }

# ==================================================
# ④ SESSION FILTER
# ==================================================
import datetime as dt

SESSIONS = {
    "ASIA":   {"start": dt.time(0,  0), "end": dt.time(8,  59), "color": "#7986cb"},
    "LONDON": {"start": dt.time(7,  0), "end": dt.time(15, 59), "color": "#4db6ac"},
    "NEW_YORK":{"start": dt.time(12, 0), "end": dt.time(20, 59), "color": "#ff8a65"},
    "OVERLAP": {"start": dt.time(12, 0), "end": dt.time(15, 59), "color": "#fff176"},
}

SESSION_SCORE = {
    "OVERLAP":  15,
    "LONDON":   10,
    "NEW_YORK":  8,
    "ASIA":      0,
    "OFF":      -10,
}

ASIA_ACTIVE = {"BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD", "XRPUSD",
               "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD"}

def session_filter(pair: str) -> dict:
    now_utc  = dt.datetime.now(dt.timezone.utc).time()
    active   = []

    for name, cfg in SESSIONS.items():
        s, e = cfg["start"], cfg["end"]
        if s <= now_utc <= e:
            active.append(name)

    if "OVERLAP" in active:
        session = "OVERLAP"
    elif "LONDON" in active and "NEW_YORK" not in active:
        session = "LONDON"
    elif "NEW_YORK" in active and "LONDON" not in active:
        session = "NEW_YORK"
    elif "ASIA" in active:
        session = "ASIA"
    else:
        session = "OFF"

    score_mod = SESSION_SCORE.get(session, 0)

    if pair == "XAUUSD" and session in ("ASIA", "OFF"):
        score_mod -= 15

    if pair in ASIA_ACTIVE and session == "ASIA":
        score_mod = max(score_mod, 0)

    tradeable = session not in ("OFF",) or pair in ASIA_ACTIVE

    return {
        "session":   session,
        "score_mod": score_mod,
        "tradeable": tradeable,
        "utc_time":  dt.datetime.now(dt.timezone.utc).strftime("%H:%M UTC"),
    }

# ==================================================
# RSI DIVERGENCE
# ==================================================
def rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 5) -> str:
    close      = df["Close"]
    price_diff = close.iloc[-1] - close.iloc[-lookback]
    rsi_diff   = rsi_series.iloc[-1] - rsi_series.iloc[-lookback]
    if price_diff < 0 and rsi_diff > 0: return "BULLISH_DIV"
    if price_diff > 0 and rsi_diff < 0: return "BEARISH_DIV"
    return "NONE"

# ==================================================
# BOS CONFIRMATION
# ==================================================
def bos_confirmation(df: pd.DataFrame, swing: int = 10) -> dict:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    swing_high = high.rolling(swing).max()
    swing_low  = low.rolling(swing).min()

    bull_bos = (
        close.iloc[-1] > swing_high.iloc[-swing - 1]
        and close.iloc[-1] > close.iloc[-2]
    )
    bear_bos = (
        close.iloc[-1] < swing_low.iloc[-swing - 1]
        and close.iloc[-1] < close.iloc[-2]
    )

    fresh_bull = any(
        close.iloc[-i] > swing_high.iloc[-swing - i]
        for i in range(1, 4)
    )
    fresh_bear = any(
        close.iloc[-i] < swing_low.iloc[-swing - i]
        for i in range(1, 4)
    )

    return {
        "bull_bos":   bool(bull_bos),
        "bear_bos":   bool(bear_bos),
        "fresh_bull": bool(fresh_bull),
        "fresh_bear": bool(fresh_bear),
        "swing_high": float(swing_high.iloc[-swing - 1]),
        "swing_low":  float(swing_low.iloc[-swing - 1]),
    }

# ==================================================
# CANDLE TREND CONFIRMATION
# ==================================================
def candle_confirmation(df: pd.DataFrame, div_type: str) -> dict:
    open_  = df["Open"]
    high   = df["High"]
    low    = df["Low"]
    close  = df["Close"]
    vol    = df["Volume"] if "Volume" in df.columns else pd.Series([0]*len(df))

    o1, h1, l1, c1 = float(open_.iloc[-1]), float(high.iloc[-1]), float(low.iloc[-1]), float(close.iloc[-1])
    o2, h2, l2, c2 = float(open_.iloc[-2]), float(high.iloc[-2]), float(low.iloc[-2]), float(close.iloc[-2])

    body1     = abs(c1 - o1)
    range1    = h1 - l1 if h1 != l1 else 0.0001
    body_pct  = body1 / range1

    mid_prev  = (o2 + c2) / 2

    avg_vol   = float(vol.rolling(20).mean().iloc[-1]) if vol.sum() > 0 else 0
    cur_vol   = float(vol.iloc[-1])
    vol_ok    = cur_vol > avg_vol * 0.9 if avg_vol > 0 else True

    if div_type == "BULLISH_DIV":
        bull_candle  = c1 > o1
        solid        = body_pct > 0.45
        above_mid    = c1 > mid_prev
        confirmed    = bull_candle and solid and above_mid
        candle_type  = "Bullish" if bull_candle else "Bearish"
        desc = (
            f"Candle: {'Hijau' if bull_candle else 'Merah'}  "
            f"Body: {body_pct*100:.0f}%  "
            f"Above mid prev: {'Ya' if above_mid else 'Tidak'}  "
            f"Vol OK: {'Ya' if vol_ok else 'Tidak'}"
        )
    else:
        bear_candle  = c1 < o1
        solid        = body_pct > 0.45
        below_mid    = c1 < mid_prev
        confirmed    = bear_candle and solid and below_mid
        candle_type  = "Bearish" if bear_candle else "Bullish"
        desc = (
            f"Candle: {'Merah' if bear_candle else 'Hijau'}  "
            f"Body: {body_pct*100:.0f}%  "
            f"Below mid prev: {'Ya' if below_mid else 'Tidak'}  "
            f"Vol OK: {'Ya' if vol_ok else 'Tidak'}"
        )

    return {
        "confirmed":   confirmed,
        "candle_type": candle_type,
        "body_pct":    body_pct,
        "vol_ok":      vol_ok,
        "desc":        desc,
    }

# ==================================================
# EMA CONFLUENCE
# ==================================================
def ema_confluence(df: pd.DataFrame):
    close = df["Close"]
    e8    = ema(close, 8).iloc[-1]
    e21   = ema(close, 21).iloc[-1]
    e50   = ema(close, 50).iloc[-1]
    e200  = ema(close, 200).iloc[-1]
    price = close.iloc[-1]
    return {
        "BULLISH_STACK": bool(price > e8 > e21 > e50),
        "BEARISH_STACK": bool(price < e8 < e21 < e50),
        "NEAR_200EMA":   bool(abs(price - e200) / e200 < 0.005),
        "E8": e8, "E21": e21, "E50": e50, "E200": e200,
    }

# ==================================================
# SMC ENGINE
# ==================================================
def smc(df: pd.DataFrame) -> dict:
    close, high, low = df["Close"], df["High"], df["Low"]
    swing_high = high.rolling(10).max()
    swing_low  = low.rolling(10).min()
    body       = (close - df["Open"]).abs()
    return {
        "BOS":         bool(close.iloc[-1] > swing_high.iloc[-2]),
        "CHoCH":       bool(close.iloc[-1] < swing_low.iloc[-2]),
        "LIQ":         bool(high.iloc[-1] > high.rolling(20).max().iloc[-2]),
        "ORDER_BLOCK": bool(body.iloc[-3] > body.rolling(10).mean().iloc[-3]),
    }

# ==================================================
# LIQUIDITY ENGINE
# ==================================================
def liquidity(df: pd.DataFrame) -> dict:
    high_zone  = df["High"].rolling(20).max().iloc[-1]
    low_zone   = df["Low"].rolling(20).min().iloc[-1]
    price      = df["Close"].iloc[-1]
    pressure   = "BUY" if abs(price - low_zone) < abs(price - high_zone) else "SELL"
    return {"HIGH_ZONE": high_zone, "LOW_ZONE": low_zone, "PRESSURE": pressure}

# ==================================================
# FLOW ENGINE
# ==================================================
def flow(df: pd.DataFrame) -> dict:
    momentum = df["Close"].diff().mean()
    return {"FLOW": "BUY" if momentum > 0 else "SELL", "STRENGTH": abs(momentum) * 100}

# ==================================================
# TREND STRENGTH
# ==================================================
def trend_strength(df: pd.DataFrame, period: int = 14) -> dict:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr_s    = atr(df, period)
    plus_di  = pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr_s * 100
    minus_di = pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr_s * 100

    dx       = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
    adx      = dx.rolling(period).mean()

    adx_val  = float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0
    pdi      = float(plus_di.iloc[-1])  if not np.isnan(plus_di.iloc[-1])  else 25.0
    mdi      = float(minus_di.iloc[-1]) if not np.isnan(minus_di.iloc[-1]) else 25.0

    trending = adx_val > 25
    bull_trend = pdi > mdi

    return {
        "adx":        adx_val,
        "plus_di":    pdi,
        "minus_di":   mdi,
        "trending":   trending,
        "bull_trend": bull_trend,
    }

# ==================================================
# LSTM PROXY
# ==================================================
def lstm_prediction(df: pd.DataFrame) -> str:
    close  = df["Close"]
    price  = close.iloc[-1]
    score  = sum([
        price > ema(close, 20).iloc[-1],
        price > ema(close, 50).iloc[-1],
        price > ema(close, 200).iloc[-1],
    ])
    return "BULLISH" if score >= 2 else "BEARISH"

# ==================================================
# ⑤ ADVANCED RULE-BASED SCORING
# ==================================================
def advanced_score(
    lstm, smc_data, liq, fl, rsi_val, macd_hist, ema_conf,
    rsi_div, vol_ratio, stoch_k_val, bb_pos,
    of: dict, fvg: dict, mtf: dict, sess: dict, trend: dict,
) -> dict:
    cats = {}

    t = 0
    if lstm == "BULLISH":           t += 10
    else:                           t -= 10
    if ema_conf["BULLISH_STACK"]:   t += 8
    elif ema_conf["BEARISH_STACK"]: t -= 8
    if ema_conf["NEAR_200EMA"]:     t += 3
    if trend["trending"]:
        t += 4 if trend["bull_trend"] else -4
    cats["TREND"] = max(-25, min(25, t))

    m = 0
    if rsi_val < 30:         m += 10
    elif rsi_val > 70:       m -= 10
    elif rsi_val > 55:       m += 4
    elif rsi_val < 45:       m -= 4
    if rsi_div == "BULLISH_DIV":  m += 6
    elif rsi_div == "BEARISH_DIV": m -= 6
    if macd_hist > 0:        m += 4
    else:                    m -= 4
    if stoch_k_val < 20:     m += 4
    elif stoch_k_val > 80:   m -= 4
    cats["MOMENTUM"] = max(-20, min(20, m))

    cats["SMC"] = smc_score_modifier(smc_data)

    o = 0
    if of["bias"] == "BUY":      o += 8
    elif of["bias"] == "SELL":   o -= 8
    if of["buy_vol_pct"] > 60:   o += 4
    elif of["buy_vol_pct"] < 40: o -= 4
    if of["absorption"]:         o += 3
    if vol_ratio >= 1.5:         o += 3
    elif vol_ratio < 0.7:        o -= 3
    cats["ORDERFLOW"] = max(-15, min(15, o))

    f = 0
    if fvg["in_bull_fvg"]:       f += 10
    elif fvg["in_bear_fvg"]:     f -= 10
    elif fvg["count_bull"] > 0:  f += 3
    elif fvg["count_bear"] > 0:  f -= 3
    cats["FVG"] = max(-10, min(10, f))

    mtf_s = 0
    if mtf["bias"] == "BULLISH":   mtf_s += 15
    elif mtf["bias"] == "BEARISH": mtf_s -= 15
    cats["MTF"] = max(-15, min(15, mtf_s))

    cats["SESSION"] = max(-10, min(10, sess["score_mod"]))

    b = 0
    if bb_pos < 0.15:   b += 5
    elif bb_pos > 0.85: b -= 5
    cats["BB"] = b

    total = 50 + sum(cats.values())
    total = max(0, min(100, total))

    direction = "BUY" if total >= 55 else "SELL"
    agree = sum([
        cats["TREND"]     > 0 if direction == "BUY" else cats["TREND"]     < 0,
        cats["MOMENTUM"]  > 0 if direction == "BUY" else cats["MOMENTUM"]  < 0,
        cats["SMC"]       > 0 if direction == "BUY" else cats["SMC"]       < 0,
        cats["ORDERFLOW"] > 0 if direction == "BUY" else cats["ORDERFLOW"] < 0,
        cats["FVG"]       > 0 if direction == "BUY" else cats["FVG"]       < 0,
        cats["MTF"]       > 0 if direction == "BUY" else cats["MTF"]       < 0,
    ])

    return {
        "total":       total,
        "cats":        cats,
        "confluence":  agree,
    }

# ==================================================
# CONFIDENCE LABEL
# ==================================================
def confidence_label(score: int, confluence: int = 0) -> str:
    if score >= 82 and confluence >= 5: return "🔥 VERY HIGH"
    if score >= 70 and confluence >= 4: return "✅ HIGH"
    if score >= 55:                      return "⚠️ MODERATE"
    return "❌ LOW"

# ==================================================
# LIQUIDITY POOL ZONES
# ==================================================
def liquidity_pools(df: pd.DataFrame, swing: int = 20) -> dict:
    high    = df["High"]
    low     = df["Low"]
    atr_now = float(df["High"].rolling(14).mean().iloc[-1] -
                    df["Low"].rolling(14).mean().iloc[-1])
    swing_low  = float(low.rolling(swing).min().iloc[-1])
    swing_high = float(high.rolling(swing).max().iloc[-1])
    buf = atr_now * 0.3
    return {
        "buy_side_liq":  swing_low,
        "sell_side_liq": swing_high,
        "buf":           buf,
        "swing":         swing,
    }

# ==================================================
# TP/SL ANTI-SWEEP
# ==================================================
def tp_sl_anti_sweep(df: pd.DataFrame, action: str, atr_val: float) -> dict:
    price = float(df["Close"].iloc[-1])
    pools = liquidity_pools(df)

    if action == "BUY":
        sl1       = price - 1.0 * atr_val
        raw_sl2   = pools["buy_side_liq"] - pools["buf"]
        sl2       = max(raw_sl2, price - 3.0 * atr_val)
        sl2       = min(sl2, sl1 - 0.1 * atr_val)
        sl1_dist  = abs(price - sl1)
        sl2_dist  = abs(price - sl2)
        tp1       = price + sl1_dist
        tp2_raw   = price + 2.5 * sl2_dist
        tp2       = min(tp2_raw, price + 2.0 * (tp1 - price) * 1.5)
        tp2       = max(tp2, tp1 + 0.5 * atr_val)
        sweep_ref = pools["buy_side_liq"]
        sweep_dir = "below"
    else:
        sl1       = price + 1.0 * atr_val
        raw_sl2   = pools["sell_side_liq"] + pools["buf"]
        sl2       = min(raw_sl2, price + 3.0 * atr_val)
        sl2       = max(sl2, sl1 + 0.1 * atr_val)
        sl1_dist  = abs(price - sl1)
        sl2_dist  = abs(price - sl2)
        tp1       = price - sl1_dist
        tp2_raw   = price - 2.5 * sl2_dist
        tp2       = max(tp2_raw, price - 2.0 * (price - tp1) * 1.5)
        tp2       = min(tp2, tp1 - 0.5 * atr_val)
        sweep_ref = pools["sell_side_liq"]
        sweep_dir = "above"

    rr1 = abs(tp1 - price) / sl1_dist if sl1_dist > 0 else 0
    rr2 = abs(tp2 - price) / sl2_dist if sl2_dist > 0 else 0

    return {
        "tp":  tp2,   "sl":  sl2,   "rr":  rr2,
        "tp1": tp1,   "tp2": tp2,
        "sl1": sl1,   "sl2": sl2,
        "sl1_dist": sl1_dist,  "sl2_dist": sl2_dist,
        "rr1": rr1,   "rr2": rr2,
        "sweep_ref":  sweep_ref,
        "sweep_dir":  sweep_dir,
        "buf":        pools["buf"],
        "buy_pool":   pools["buy_side_liq"],
        "sell_pool":  pools["sell_side_liq"],
    }

# ==================================================
# GENERATE SIGNAL  — UPGRADED with Regime + ML
# ==================================================
def generate_signal(pair: str, timeframe: str = DEFAULT_TF) -> dict:
    df     = get_data(pair, timeframe)
    close  = df["Close"]

    # ── base indicators ───────────────────────────
    rsi_series            = rsi(close)
    rsi_val               = float(rsi_series.iloc[-1])
    _, _, macd_h          = macd(close)
    macd_hist_val         = float(macd_h.iloc[-1])
    atr_val               = float(atr(df).iloc[-1])
    stoch_k, _            = stochastic(df)
    stoch_k_val           = float(stoch_k.iloc[-1])
    bb_upper, _, bb_lower = bollinger_bands(close)
    bb_range              = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    bb_pos                = float(
        (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range if bb_range else 0.5
    )
    vol_ratio_val  = volume_ratio(df)
    rsi_div        = rsi_divergence(df, rsi_series)
    ema_conf       = ema_confluence(df)
    smc_data       = smc_full(df)
    liq            = liquidity(df)
    fl             = flow(df)
    lstm           = lstm_prediction(df)
    trend          = trend_strength(df)

    # ── new engines ───────────────────────────────
    of             = delta_orderflow(df)
    fvg            = fvg_engine(df)
    sess           = session_filter(pair)
    mtf            = mtf_bias(pair, timeframe)

    # ── rule-based scoring ────────────────────────
    sc             = advanced_score(
        lstm, smc_data, liq, fl, rsi_val, macd_hist_val,
        ema_conf, rsi_div, vol_ratio_val, stoch_k_val, bb_pos,
        of, fvg, mtf, sess, trend,
    )

    # ── [NEW] Regime Detection ────────────────────
    regime_data = detect_regime(df)

    # First-pass action for regime modifier calculation
    reg_mod     = regime_score_modifier(regime_data, "BUY")
    rule_score  = max(0, min(100, sc["total"] + reg_mod))
    action      = "BUY" if rule_score >= 55 else "SELL"

    # Recalculate with correct action direction
    reg_mod     = regime_score_modifier(regime_data, action)
    rule_score  = max(0, min(100, sc["total"] + reg_mod))
    action      = "BUY" if rule_score >= 55 else "SELL"

    # ── [NEW] ML Adaptive Scoring ─────────────────
    _signal_draft = {
        "pair":       pair,
        "timeframe":  timeframe,
        "score":      rule_score,
        "rule_score": rule_score,
        "cats":       sc["cats"],
        "rsi":        rsi_val,
        "stoch_k":    stoch_k_val,
        "bb_pos":     bb_pos,
        "vol_ratio":  vol_ratio_val,
        "trend":      trend,
        "action":     action,
        "ema":        ema_conf,
        "orderflow":  of,
        "mtf":        mtf,
    }
    ml_result = adaptive_score(rule_score, _signal_draft)
    score     = ml_result["adaptive_score"]
    action    = "BUY" if score >= 55 else "SELL"

    sl_data   = tp_sl_anti_sweep(df, action, atr_val)
    price     = float(close.iloc[-1])

    result = {
        "pair":       pair,
        "timeframe":  timeframe,
        "score":      score,
        "rule_score": rule_score,
        "cats":       sc["cats"],
        "confluence": sc["confluence"],
        "confidence": confidence_label(score, sc["confluence"]),
        "action":     action,
        "entry":      price,
        "tp":         sl_data["tp2"],
        "sl":         sl_data["sl2"],
        "tp1":        sl_data["tp1"],
        "tp2":        sl_data["tp2"],
        "sl1":        sl_data["sl1"],
        "sl2":        sl_data["sl2"],
        "rr1":        sl_data["rr1"],
        "rr2":        sl_data["rr2"],
        "rr":         sl_data["rr2"],
        "rsi":        rsi_val,
        "macd":       macd_hist_val,
        "atr":        atr_val,
        "stoch_k":    stoch_k_val,
        "bb_pos":     bb_pos,
        "vol_ratio":  vol_ratio_val,
        "rsi_div":    rsi_div,
        "ema":        ema_conf,
        "lstm":       lstm,
        "smc":        smc_data,
        "liq":        liq,
        "flow":       fl,
        "trend":      trend,
        "orderflow":  of,
        "fvg":        fvg,
        "session":    sess,
        "mtf":        mtf,
        "sweep":      sl_data,
        "regime":     regime_data,
        "ml":         ml_result,
        "df":         df,
    }

    # ── [NEW] Persist to DB + background retrain ──
    try:
        log_signal(result)
        log_score_history(result)

       rows_snapshot = get_training_data()
if rows_snapshot:
    def _bg_retrain(rows=rows_snapshot):
        maybe_retrain(rows)
    threading.Thread(target=_bg_retrain, daemon=True).start()

    except Exception as e:
        print(f"[DB LOG ERROR] {e}")

    return result


# ==================================================
# FORMAT SIGNAL (Discord Embed)  — UPGRADED
# ==================================================
def build_embed(data: dict) -> discord.Embed:
    action_color = discord.Color.green() if data["action"] == "BUY" else discord.Color.red()
    action_emoji = "🟢" if data["action"] == "BUY" else "🔴"
    sw   = data["sweep"]
    of   = data["orderflow"]
    fvg  = data["fvg"]
    mtf  = data["mtf"]
    sess = data["session"]
    cats = data["cats"]
    tr   = data["trend"]

    # Rule score vs adaptive score display
    rule_s = data.get("rule_score", data["score"])
    ml_tag = (
        f" _(rule: {rule_s:.0f})_"
        if abs(data["score"] - rule_s) >= 1 else ""
    )

    embed = discord.Embed(
        title = (
            f"{action_emoji} {data['pair']} — {data['action']} SIGNAL  "
            f"[{data['confluence']}/6 confluence]"
        ),
        description = (
            f"**AI Score:** `{data['score']:.1f}%`{ml_tag} — {data['confidence']}\n"
            f"**Session:** `{sess['session']}`  `{sess['utc_time']}`  "
            f"{'✅ Tradeable' if sess['tradeable'] else '⚠️ Off-session'}"
        ),
        color = action_color,
    )

    # ── Entry / TP / SL ──────────────────────────
    embed.add_field(
        name  = "📍 Entry  |  🎯 TP Levels  |  🛑 SL Levels",
        value = (
            f"**Entry:** `{data['entry']:.4f}`\n"
            f"{'─'*32}\n"
            f"🟢 **TP1** `{data['tp1']:.4f}`  →  RR `1:{data['rr1']:.2f}`  *(quick, 1:1 SL1)*\n"
            f"🟩 **TP2** `{data['tp2']:.4f}`  →  RR `1:{data['rr2']:.2f}`  *(target, anti-sweep)*\n"
            f"{'─'*32}\n"
            f"🟡 **SL1** `{data['sl1']:.4f}`  →  `{data['atr']:.4f}` ATR  *(tight stop)*\n"
            f"🔴 **SL2** `{data['sl2']:.4f}`  →  anti-sweep  *(beyond liq pool)*"
        ),
        inline=False,
    )

    # ── Score breakdown ───────────────────────────
    def bar(v, mx):
        filled = int(round(abs(v) / mx * 5))
        ch = "🟩" if v >= 0 else "🟥"
        return ch * filled + "⬜" * (5 - filled)

    embed.add_field(
        name  = "📊 Score Breakdown",
        value = (
            f"Trend      {bar(cats['TREND'],     25)} `{cats['TREND']:+d}`\n"
            f"Momentum   {bar(cats['MOMENTUM'],  20)} `{cats['MOMENTUM']:+d}`\n"
            f"SMC        {bar(cats['SMC'],        20)} `{cats['SMC']:+d}`\n"
            f"Orderflow  {bar(cats['ORDERFLOW'], 15)} `{cats['ORDERFLOW']:+d}`\n"
            f"FVG        {bar(cats['FVG'],        10)} `{cats['FVG']:+d}`\n"
            f"MTF Bias   {bar(cats['MTF'],        15)} `{cats['MTF']:+d}`\n"
            f"Session    {bar(cats['SESSION'],    10)} `{cats['SESSION']:+d}`"
        ),
        inline=False,
    )

    # ── MTF Bias ─────────────────────────────────
    htf_lines = []
    for tf, d in mtf["detail"].items():
        if "error" not in d:
            emoji = "🟢" if d["bias"] == "BULLISH" else "🔴" if d["bias"] == "BEARISH" else "⚪"
            htf_lines.append(f"{emoji} `{tf.upper()}` {d['bias']}  RSI:`{d['rsi']:.0f}`")
    embed.add_field(
        name  = "🕐 MTF Bias",
        value = "\n".join(htf_lines) if htf_lines else "N/A",
        inline=True,
    )

    # ── Orderflow Delta ───────────────────────────
    of_emoji = "🟢" if of["bias"] == "BUY" else "🔴" if of["bias"] == "SELL" else "⚪"
    embed.add_field(
        name  = "📦 Delta Orderflow",
        value = (
            f"{of_emoji} Bias: `{of['bias']}`\n"
            f"Buy Vol: `{of['buy_vol_pct']:.1f}%`\n"
            f"CVD Slope: `{of['cvd_slope']:.2f}`\n"
            f"Absorption: `{'YES ⚡' if of['absorption'] else 'NO'}`"
        ),
        inline=True,
    )

    # ── FVG ──────────────────────────────────────
    fvg_status = "🟩 IN BULL FVG" if fvg["in_bull_fvg"] else "🟥 IN BEAR FVG" if fvg["in_bear_fvg"] else "➖ Outside FVG"
    nb = fvg["nearest_bull"]
    nd = fvg["nearest_bear"]
    embed.add_field(
        name  = "📐 FVG / Imbalance",
        value = (
            f"{fvg_status}\n"
            f"Bull FVGs: `{fvg['count_bull']}`  Bear: `{fvg['count_bear']}`\n"
            + (f"Nearest Bull: `{nb['bottom']:.4f}`-`{nb['top']:.4f}`\n" if nb else "")
            + (f"Nearest Bear: `{nd['bottom']:.4f}`-`{nd['top']:.4f}`" if nd else "")
        ),
        inline=False,
    )

    # ── Indicators ───────────────────────────────
    div_text = {"BULLISH_DIV": "🔼 Bullish", "BEARISH_DIV": "🔽 Bearish", "NONE": "—"}
    embed.add_field(
        name  = "📊 Indicators",
        value = (
            f"RSI: `{data['rsi']:.1f}` | MACD H: `{data['macd']:.4f}`\n"
            f"Stoch K: `{data['stoch_k']:.1f}` | BB Pos: `{data['bb_pos']:.2f}`\n"
            f"Vol Ratio: `{data['vol_ratio']:.2f}x` | ADX: `{tr['adx']:.1f}`\n"
            f"RSI Div: {div_text[data['rsi_div']]}"
        ),
        inline=True,
    )

    # ── SMC / Structure ──────────────────────────
    ema_lbl = "🐂 Bull Stack" if data["ema"]["BULLISH_STACK"] else "🐻 Bear Stack" if data["ema"]["BEARISH_STACK"] else "Mixed"
    embed.add_field(
        name  = "🏗 SMC / Structure",
        value = (
            f"{smc_embed_value(data['smc'])}\n"
            f"EMA: {ema_lbl}\n"
            f"LSTM: `{data['lstm']}` | Trend: `{'Strong' if tr['trending'] else 'Weak'}`\n"
            f"Liq Pressure: `{data['liq']['PRESSURE']}`"
        ),
        inline=False,
    )

    # ── Anti-Sweep SL ─────────────────────────────
    sweep_pos  = "di bawah" if data["action"] == "BUY" else "di atas"
    embed.add_field(
        name  = "🛡️ Anti Liquidity Sweep SL",
        value = (
            f"SL1 `{data['sl1']:.4f}` ← 1×ATR *(tight, trail ke sini setelah TP1 hit)*\n"
            f"SL2 `{data['sl2']:.4f}` ← {sweep_pos} pool `{sw['sweep_ref']:.4f}` + buf `{sw['buf']:.4f}`\n"
            f"Buy Pool: `{sw['buy_pool']:.4f}` | Sell Pool: `{sw['sell_pool']:.4f}`"
        ),
        inline=False,
    )

    # ── [NEW] Regime Detection ────────────────────
    if "regime" in data:
        skip_flag, skip_reason = should_skip(data["regime"], data["action"], data["score"])
        regime_val = regime_embed_value(data["regime"])
        if skip_flag:
            regime_val += f"\n⚠️ **SKIP ALERT:** {skip_reason}"
        embed.add_field(
            name   = f"{data['regime']['emoji']} Market Regime",
            value  = regime_val,
            inline = False,
        )

    # ── [NEW] ML Adaptive Score ───────────────────
    if "ml" in data:
        embed.add_field(
            name   = "🤖 ML Adaptive Score",
            value  = ml_embed_value(data["ml"]),
            inline = False,
        )

    embed.set_footer(
        text=f"⏱ TF: {data['timeframe'].upper()} | MTF: {'+'.join(mtf['htf_list'])} | APC-Institional Engine by Yor v4"
    )
    return embed


# ==================================================
# CHART
# ==================================================
def create_chart(
    df: pd.DataFrame,
    pair: str,
    entry: float,
    tp: float,
    sl: float,
    timeframe: str = DEFAULT_TF,
    sweep_data: dict = None,
    fvg_data:   dict = None,
    tp1: float  = None,
    sl1: float  = None,
    rr1: float  = 0.0,
    rr2: float  = 0.0,
) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    df_c = df.copy()

    if not isinstance(df_c.index, pd.DatetimeIndex):
        df_c.index = pd.to_datetime(df_c.index)

    candle_limit = {"5m": 100, "15m": 100, "1h": 96, "4h": 60}
    n    = candle_limit.get(timeframe, 100)
    df_c = df_c.iloc[-n:]

    close = df_c["Close"]
    df_c["EMA8"]  = ema(close, 8)
    df_c["EMA21"] = ema(close, 21)
    df_c["EMA50"] = ema(close, 50)

    xs     = np.arange(len(df_c))
    opens  = df_c["Open"].values
    highs  = df_c["High"].values
    lows   = df_c["Low"].values
    closes = df_c["Close"].values

    is_buy      = entry < tp
    action_lbl  = "BUY" if is_buy else "SELL"

    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = fig.add_gridspec(
        2, 1, height_ratios=[5, 1],
        hspace=0.06, left=0.01, right=0.86, top=0.93, bottom=0.08,
    )
    ax  = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")
    ax2.axis("off")

    if sweep_data:
        buy_pool  = sweep_data["buy_pool"]
        sell_pool = sweep_data["sell_pool"]
        buf       = sweep_data["buf"]
        ax.axhspan(buy_pool - buf * 2, buy_pool, alpha=0.08, color="#ffd600", zorder=1)
        ax.axhline(buy_pool, color="#ffd600", linewidth=0.9, linestyle=":", zorder=2, alpha=0.7)
        ax.axhspan(sell_pool, sell_pool + buf * 2, alpha=0.08, color="#ce93d8", zorder=1)
        ax.axhline(sell_pool, color="#ce93d8", linewidth=0.9, linestyle=":", zorder=2, alpha=0.7)
        ax.text(0.2, buy_pool, f" BUY POOL  {buy_pool:.4f}", color="#ffd600", fontsize=7.5, va="bottom", alpha=0.85)
        ax.text(0.2, sell_pool, f" SELL POOL  {sell_pool:.4f}", color="#ce93d8", fontsize=7.5, va="top", alpha=0.85)

    if is_buy:
        ax.axhspan(entry, tp,  alpha=0.08, color="#00c853", zorder=1)
        if tp1: ax.axhspan(entry, tp1, alpha=0.14, color="#69f0ae", zorder=1)
    else:
        ax.axhspan(tp,  entry, alpha=0.08, color="#00c853", zorder=1)
        if tp1: ax.axhspan(tp1, entry, alpha=0.14, color="#69f0ae", zorder=1)

    if is_buy:
        ax.axhspan(sl, entry,  alpha=0.10, color="#d50000", zorder=1)
        if sl1: ax.axhspan(sl1, entry, alpha=0.14, color="#ff6d00", zorder=1)
    else:
        ax.axhspan(entry, sl,  alpha=0.10, color="#d50000", zorder=1)
        if sl1: ax.axhspan(entry, sl1, alpha=0.14, color="#ff6d00", zorder=1)

    if sweep_data:
        sweep_ref = sweep_data["sweep_ref"]
        if is_buy:
            ax.axhspan(sl, sweep_ref, alpha=0.20, color="#b71c1c", zorder=1)
            ax.axhline(sweep_ref, color="#ff6d00", linewidth=0.9, linestyle="-.", zorder=5, alpha=0.8)
            ax.text(xs[-1] + 0.5, sweep_ref, f" POOL  {sweep_ref:.4f}", color="#ff6d00", fontsize=7, va="center", fontweight="bold")
        else:
            ax.axhspan(sweep_ref, sl, alpha=0.20, color="#b71c1c", zorder=1)
            ax.axhline(sweep_ref, color="#ff6d00", linewidth=0.9, linestyle="-.", zorder=5, alpha=0.8)
            ax.text(xs[-1] + 0.5, sweep_ref, f" POOL  {sweep_ref:.4f}", color="#ff6d00", fontsize=7, va="center", fontweight="bold")

        ax.axhspan(sweep_data["buy_pool"] - sweep_data["buf"]*2, sweep_data["buy_pool"], alpha=0.07, color="#ffd600", zorder=1)
        ax.axhline(sweep_data["buy_pool"], color="#ffd600", linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)
        ax.axhspan(sweep_data["sell_pool"], sweep_data["sell_pool"] + sweep_data["buf"]*2, alpha=0.07, color="#ce93d8", zorder=1)
        ax.axhline(sweep_data["sell_pool"], color="#ce93d8", linewidth=0.8, linestyle=":", alpha=0.6, zorder=2)

    w = 0.6
    for i, x in enumerate(xs):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([x, x], [l, h], color=col, linewidth=0.8, zorder=6)
        body_lo = min(o, c)
        body_hi = max(o, c)
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, body_lo), w,
            max(body_hi - body_lo, (h - l) * 0.003),
            boxstyle="square,pad=0", linewidth=0, facecolor=col, zorder=7,
        )
        ax.add_patch(rect)

    if fvg_data:
        for fv in fvg_data["bull_fvgs"][-3:]:
            ax.axhspan(fv["bottom"], fv["top"], alpha=0.10, color="#00bcd4", zorder=2)
            ax.axhline(fv["top"],    color="#00bcd4", linewidth=0.6, linestyle=":", alpha=0.6, zorder=3)
            ax.axhline(fv["bottom"], color="#00bcd4", linewidth=0.6, linestyle=":", alpha=0.6, zorder=3)
        for fv in fvg_data["bear_fvgs"][-3:]:
            ax.axhspan(fv["bottom"], fv["top"], alpha=0.10, color="#e040fb", zorder=2)
            ax.axhline(fv["top"],    color="#e040fb", linewidth=0.6, linestyle=":", alpha=0.6, zorder=3)
            ax.axhline(fv["bottom"], color="#e040fb", linewidth=0.6, linestyle=":", alpha=0.6, zorder=3)
        nb = fvg_data["nearest_bull"]
        nd = fvg_data["nearest_bear"]
        if nb:
            ax.text(0.5, (nb["top"] + nb["bottom"]) / 2, f"  BULL FVG {nb['bottom']:.4f}-{nb['top']:.4f}", color="#00bcd4", fontsize=6.5, va="center", alpha=0.85)
        if nd:
            ax.text(0.5, (nd["top"] + nd["bottom"]) / 2, f"  BEAR FVG {nd['bottom']:.4f}-{nd['top']:.4f}", color="#e040fb", fontsize=6.5, va="center", alpha=0.85)

    ax.plot(xs, df_c["EMA8"].values,  color="#00e5ff", linewidth=1.0, label="EMA 8",  zorder=8)
    ax.plot(xs, df_c["EMA21"].values, color="#ffeb3b", linewidth=1.0, label="EMA 21", zorder=8)
    ax.plot(xs, df_c["EMA50"].values, color="#ff9800", linewidth=1.2, label="EMA 50", zorder=8)

    lx = xs[-1] + 0.5
    ax.axhline(tp, color="#00e676", linewidth=1.8, linestyle="--", zorder=9)
    ax.text(lx, tp, f" [TP2]  {tp:.4f}  RR 1:{rr2:.2f}", color="#00e676", fontsize=8, va="center", fontweight="bold")
    if tp1:
        ax.axhline(tp1, color="#b9f6ca", linewidth=1.3, linestyle="-.", zorder=9)
        ax.text(lx, tp1, f" [TP1]  {tp1:.4f}  RR 1:{rr1:.2f}", color="#b9f6ca", fontsize=8, va="center", fontweight="bold")
    ax.axhline(entry, color="#ffffff", linewidth=1.1, linestyle="--", zorder=9, alpha=0.85)
    ax.text(lx, entry, f" ENTRY  {entry:.4f}", color="#ffffff", fontsize=8, va="center", fontweight="bold", alpha=0.9)
    if sl1:
        ax.axhline(sl1, color="#ffab40", linewidth=1.3, linestyle="-.", zorder=9)
        ax.text(lx, sl1, f" [SL1]  {sl1:.4f}  1xATR", color="#ffab40", fontsize=8, va="center", fontweight="bold")
    ax.axhline(sl, color="#ff1744", linewidth=1.8, linestyle="--", zorder=9)
    ax.text(lx, sl, f" [SL2]  {sl:.4f}  anti-sweep", color="#ff1744", fontsize=8, va="center", fontweight="bold")

    bx1 = -0.8
    bx2 = -2.0
    if tp1 and sl1:
        ax.annotate("", xy=(bx1, tp1), xytext=(bx1, entry), arrowprops=dict(arrowstyle="<->", color="#b9f6ca", lw=1.0))
        ax.text(bx1 - 0.2, (entry + tp1) / 2, "T1", color="#b9f6ca", fontsize=6.5, ha="right", va="center")
        ax.annotate("", xy=(bx1, sl1), xytext=(bx1, entry), arrowprops=dict(arrowstyle="<->", color="#ffab40", lw=1.0))
        ax.text(bx1 - 0.2, (entry + sl1) / 2, "S1", color="#ffab40", fontsize=6.5, ha="right", va="center")
    ax.annotate("", xy=(bx2, tp), xytext=(bx2, entry), arrowprops=dict(arrowstyle="<->", color="#00e676", lw=1.2))
    ax.text(bx2 - 0.2, (entry + tp) / 2, "T2", color="#00e676", fontsize=6.5, ha="right", va="center")
    ax.annotate("", xy=(bx2, sl), xytext=(bx2, entry), arrowprops=dict(arrowstyle="<->", color="#ff1744", lw=1.2))
    ax.text(bx2 - 0.2, (entry + sl) / 2, "S2", color="#ff1744", fontsize=6.5, ha="right", va="center")

    tick_every  = max(1, len(xs) // 10)
    tick_idx    = xs[::tick_every]
    tick_labels = [df_c.index[i].strftime("%m/%d %H:%M") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7, color="#aaaaaa")

    all_levels = [min(lows), max(highs), tp, sl]
    if tp1:  all_levels.append(tp1)
    if sl1:  all_levels.append(sl1)
    if sweep_data:
        all_levels += [sweep_data["buy_pool"], sweep_data["sell_pool"]]
    y_lo = min(all_levels)
    y_hi = max(all_levels)
    pad  = (y_hi - y_lo) * 0.07
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlim(-2.8, xs[-1] + 15)

    ax.yaxis.set_tick_params(labelcolor="#aaaaaa", labelsize=8)
    ax.yaxis.tick_right()
    ax.grid(axis="y", color="#1f2937", linewidth=0.5, linestyle="-")
    ax.grid(axis="x", color="#1f2937", linewidth=0.3, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")

    legend_items = [
        Line2D([0], [0], color="#00e5ff", lw=1.2, label="EMA 8"),
        Line2D([0], [0], color="#ffeb3b", lw=1.2, label="EMA 21"),
        Line2D([0], [0], color="#ff9800", lw=1.4, label="EMA 50"),
        mpatches.Patch(facecolor="#69f0ae", alpha=0.5, label="TP1 Zone"),
        mpatches.Patch(facecolor="#00c853", alpha=0.3, label="TP2 Zone"),
        mpatches.Patch(facecolor="#ff6d00", alpha=0.4, label="SL1 Zone"),
        mpatches.Patch(facecolor="#d50000", alpha=0.3, label="SL2 Zone"),
        mpatches.Patch(facecolor="#00bcd4", alpha=0.3, label="Bull FVG"),
        mpatches.Patch(facecolor="#e040fb", alpha=0.3, label="Bear FVG"),
    ]
    if sweep_data:
        legend_items += [
            Line2D([0], [0], color="#ffd600", lw=1, linestyle=":", label="Buy-Side Pool"),
            Line2D([0], [0], color="#ce93d8", lw=1, linestyle=":", label="Sell-Side Pool"),
            Line2D([0], [0], color="#ff6d00", lw=1, linestyle="-.", label="Liq Sweep Ref"),
            mpatches.Patch(facecolor="#b71c1c", alpha=0.5, label="Anti-Sweep Buffer"),
        ]
    ax.legend(handles=legend_items, loc="upper left", fontsize=7.5, facecolor="#161b22",
              edgecolor="#30363d", labelcolor="#cccccc", ncol=2, framealpha=0.85)

    rr_val = rr2 if rr2 else (abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0)
    title_color = "#00e676" if is_buy else "#ff1744"
    fig.suptitle(
        f"{pair}  ·  {timeframe.upper()}  ·  {action_lbl}  @  {entry:.4f}"
        f"   |   TP1:{rr1:.1f}  TP2 RR 1:{rr2:.2f}",
        color=title_color, fontsize=13, fontweight="bold", y=0.97,
    )

    if sweep_data:
        tp1_str = f"{tp1:.4f}" if tp1 else "-"
        sl1_str = f"{sl1:.4f}" if sl1 else "-"
        sweep_txt = (
            f"[TP1] {tp1_str} (RR 1:{rr1:.2f})  |  "
            f"[TP2] {tp:.4f} (RR 1:{rr2:.2f})  ||  "
            f"[SL1] {sl1_str} tight  |  "
            f"[SL2] {sl:.4f} anti-sweep  pool@{sweep_data['sweep_ref']:.4f}"
        )
    else:
        tp1_str = f"{tp1:.4f}" if tp1 else "-"
        sl1_str = f"{sl1:.4f}" if sl1 else "-"
        sweep_txt = (
            f"[TP1] {tp1_str}  |  [TP2] {tp:.4f}  ||  "
            f"[SL1] {sl1_str}  |  [SL2] {sl:.4f}"
        )
    ax2.text(
        0.5, 0.5, sweep_txt,
        transform=ax2.transAxes,
        color="#e0e0e0", fontsize=8.5, ha="center", va="center",
        bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round,pad=0.4"),
    )

    ax.text(0.5, 0.5, "APC-Institional Engine by Yor", transform=ax.transAxes,
            fontsize=30, color="white", alpha=0.03,
            ha="center", va="center", rotation=30, fontweight="bold")

    filename = f"{pair}_candles.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename

# ==================================================
# UI VIEWS
# ==================================================

class PairSelectView(View):
    def __init__(self, timeframe: str = DEFAULT_TF, mode: str = "signal"):
        super().__init__(timeout=60)
        self.timeframe = timeframe
        self.mode      = mode

        options = [
            discord.SelectOption(label=p, description=PAIRS[p]["yf"], emoji="📈")
            for p in PAIRS
        ]
        select = Select(placeholder="Pilih pair...", options=options)
        select.callback = self.on_pair_select
        self.add_item(select)

    async def on_pair_select(self, interaction: discord.Interaction):
        pair = interaction.data["values"][0]
        await interaction.response.defer()
        await send_signal_or_chart(interaction, pair, self.timeframe, self.mode)


class TimeframeSelectView(View):
    def __init__(self, pair: str, mode: str = "signal", channel_id: int = None):
        super().__init__(timeout=60)
        self.pair       = pair
        self.mode       = mode
        self.channel_id = channel_id

        options = [
            discord.SelectOption(label=tf.upper(), value=tf, description=f"Interval {tf}")
            for tf in TIMEFRAMES
        ]
        select = Select(placeholder="Pilih timeframe...", options=options)
        select.callback = self.on_tf_select
        self.add_item(select)

    async def on_tf_select(self, interaction: discord.Interaction):
        tf = interaction.data["values"][0]
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(
            f"⏳ Memproses `{self.pair}` `{tf.upper()}`...", ephemeral=True
        )

        channel = (
            interaction.channel
            if self.channel_id is None
            else interaction.client.get_channel(self.channel_id)
        )
        try:
            data  = await run_blocking(generate_signal, self.pair, tf)
            embed = build_embed(data)
            view  = SignalActionView(data)
            if self.mode == "chart":
                path = await run_blocking(
                    create_chart, data["df"], self.pair, data["entry"], data["tp"], data["sl"],
                    tf, data["sweep"], data["fvg"], data["tp1"], data["sl1"], data["rr1"], data["rr2"]
                )
                await channel.send(f"📊 **{self.pair}** `{tf.upper()}` Candle Chart", file=discord.File(path))
            else:
                await channel.send(embed=embed, view=view)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"TF SELECT ERROR [{self.pair}|{tf}]:\n{tb}")
            await channel.send(f"❌ Error `{self.pair}` `{tf}`: `{e}`")


class SignalActionView(View):
    def __init__(self, data: dict):
        super().__init__(timeout=120)
        self.data = data

    @discord.ui.button(label="📊 Candle Chart", style=discord.ButtonStyle.primary)
    async def show_chart(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        d = self.data
        try:
            path = await run_blocking(
                create_chart, d["df"], d["pair"], d["entry"], d["tp"], d["sl"],
                d["timeframe"], d["sweep"], d["fvg"], d["tp1"], d["sl1"], d["rr1"], d["rr2"]
            )
            await interaction.followup.send(
                f"📊 **{d['pair']}** `{d['timeframe'].upper()}` Candle Chart",
                file=discord.File(path),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"CHART ERROR:\n{tb}")
            await interaction.followup.send(f"❌ Gagal buat chart: `{e}`")

    @discord.ui.button(label="🔄 Rescan Pair Ini", style=discord.ButtonStyle.secondary)
    async def rescan(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        try:
            new_data = await run_blocking(generate_signal, self.data["pair"], self.data["timeframe"])
            embed    = build_embed(new_data)
            view     = SignalActionView(new_data)
            await interaction.followup.send(embed=embed, view=view)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"RESCAN ERROR:\n{tb}")
            await interaction.followup.send(f"❌ Gagal rescan: `{e}`")

    @discord.ui.button(label="📋 Ganti Timeframe", style=discord.ButtonStyle.secondary)
    async def change_tf(self, interaction: discord.Interaction, button: Button):
        view = TimeframeSelectView(self.data["pair"], mode="signal", channel_id=interaction.channel_id)
        await interaction.response.send_message("⏱ Pilih timeframe:", view=view, ephemeral=True)

    @discord.ui.button(label="🏆 Best Signal Semua Pair", style=discord.ButtonStyle.success)
    async def best_signal(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await interaction.followup.send(f"⏳ Scanning semua pair `{self.data['timeframe'].upper()}`...")
        await scan_best_and_send(interaction, self.data["timeframe"])


class MainMenuView(View):
    def __init__(self):
        super().__init__(timeout=120)

    @discord.ui.button(label="🏆 Best Signal", style=discord.ButtonStyle.success, row=0)
    async def best_signal(self, interaction: discord.Interaction, button: Button):
        view = TimeframeSelectMenu(mode="best")
        await interaction.response.send_message("⏱ Pilih timeframe:", view=view, ephemeral=True)

    @discord.ui.button(label="📈 Signal per Pair", style=discord.ButtonStyle.primary, row=0)
    async def signal_pair(self, interaction: discord.Interaction, button: Button):
        view = PairSelectView(mode="signal")
        await interaction.response.send_message("📈 Pilih pair:", view=view, ephemeral=True)

    @discord.ui.button(label="📊 Chart per Pair", style=discord.ButtonStyle.primary, row=0)
    async def chart_pair(self, interaction: discord.Interaction, button: Button):
        view = PairSelectView(mode="chart")
        await interaction.response.send_message("📊 Pilih pair untuk chart:", view=view, ephemeral=True)

    @discord.ui.button(label="🔁 Scan Semua Pair", style=discord.ButtonStyle.secondary, row=1)
    async def scan_all(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await interaction.followup.send("⏳ Scanning semua pair...")
        await scan_all_pairs_and_send(interaction, DEFAULT_TF)

    @discord.ui.button(label="⚙️ Auto Signal Toggle", style=discord.ButtonStyle.danger, row=1)
    async def toggle_auto(self, interaction: discord.Interaction, button: Button):
        global AUTO_SIGNAL
        AUTO_SIGNAL = not AUTO_SIGNAL
        status = "🟢 AKTIF" if AUTO_SIGNAL else "🔴 MATI"
        await interaction.response.send_message(f"Auto Signal sekarang: **{status}**", ephemeral=True)


class TimeframeSelectMenu(View):
    def __init__(self, mode: str = "best"):
        super().__init__(timeout=60)
        self.mode = mode
        options = [discord.SelectOption(label=tf.upper(), value=tf) for tf in TIMEFRAMES]
        sel = Select(placeholder="Pilih timeframe...", options=options)
        sel.callback = self.on_select
        self.add_item(sel)

    async def on_select(self, interaction: discord.Interaction):
        tf = interaction.data["values"][0]
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(f"⏳ Mencari best signal `{tf.upper()}`...", ephemeral=True)
        channel = interaction.channel

        async def _safe(pair):
            try:
                return await run_blocking(generate_signal, pair, tf)
            except Exception as e:
                print(f"TF MENU ERROR {pair}: {e}")
                return None

        results = await asyncio.gather(*[_safe(p) for p in PAIRS])
        results = [r for r in results if r is not None]

        if not results:
            await channel.send("❌ Tidak ada signal tersedia.")
            return

        best  = max(results, key=lambda x: x["score"])
        path  = await run_blocking(
            create_chart, best["df"], best["pair"], best["entry"], best["tp"], best["sl"],
            best["timeframe"], best["sweep"], best["fvg"], best["tp1"], best["sl1"], best["rr1"], best["rr2"]
        )
        embed = build_embed(best)
        view  = SignalActionView(best)
        await channel.send(embed=embed, view=view, file=discord.File(path))


# ==================================================
# HELPER: send signal or chart
# ==================================================
async def send_signal_or_chart(interaction, pair, timeframe, mode):
    try:
        data = await run_blocking(generate_signal, pair, timeframe)
        if mode == "chart":
            path = await run_blocking(
                create_chart, data["df"], pair, data["entry"], data["tp"], data["sl"],
                data["timeframe"], data["sweep"], data["fvg"], data["tp1"], data["sl1"], data["rr1"], data["rr2"]
            )
            await interaction.followup.send(f"📊 **{pair}** `{timeframe.upper()}` Candle Chart", file=discord.File(path))
        else:
            embed = build_embed(data)
            view  = SignalActionView(data)
            await interaction.followup.send(embed=embed, view=view)
    except Exception as e:
        tb = traceback.format_exc()
        print(f"send_signal_or_chart ERROR [{pair}|{timeframe}]:\n{tb}")
        await interaction.followup.send(f"❌ Error memproses {pair}: `{e}`")


async def scan_best_and_send(interaction, timeframe):
    sem = _get_tv_semaphore()

    async def _safe_signal(pair):
        async with sem:
            try:
                return await run_blocking(generate_signal, pair, timeframe)
            except Exception as e:
                print(f"SCAN ERROR {pair}: {e}")
                return None

    results = await asyncio.gather(*[_safe_signal(p) for p in PAIRS])
    results = [r for r in results if r is not None]

    if not results:
        await interaction.followup.send("❌ Tidak ada signal tersedia.")
        return

    best  = max(results, key=lambda x: x["score"])
    path  = await run_blocking(
        create_chart, best["df"], best["pair"], best["entry"], best["tp"], best["sl"],
        best["timeframe"], best["sweep"], best["fvg"], best["tp1"], best["sl1"], best["rr1"], best["rr2"]
    )
    embed = build_embed(best)
    view  = SignalActionView(best)
    await interaction.followup.send(embed=embed, view=view, file=discord.File(path))


async def scan_all_pairs_and_send(interaction, timeframe):
    sem = _get_tv_semaphore()

    async def _safe_signal(pair):
        async with sem:
            try:
                return pair, await run_blocking(generate_signal, pair, timeframe), None
            except Exception as e:
                return pair, None, str(e)

    results = await asyncio.gather(*[_safe_signal(p) for p in PAIRS])

    lines = []
    for pair, d, err in results:
        if err:
            lines.append(f"⚠️ **{pair}** error: {err}")
        else:
            emoji = "🟢" if d["action"] == "BUY" else "🔴"
            reg_emoji = d["regime"]["emoji"] if "regime" in d else ""
            lines.append(
                f"{emoji} **{pair}** | Score: `{d['score']:.1f}%` | {d['confidence']}"
                f" | Entry: `{d['entry']:.4f}` {reg_emoji}"
            )

    embed = discord.Embed(
        title       = f"🔁 Scan Semua Pair — {timeframe.upper()}",
        description = "\n".join(lines),
        color       = discord.Color.blurple(),
    )
    embed.set_footer(text="Klik '📈 Signal per Pair' untuk detail sinyal per pair")
    await interaction.followup.send(embed=embed)

# ==================================================
# AUTOCOMPLETE FUNCTIONS
# ==================================================

async def autocomplete_pair(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Dropdown pair — filter by current input."""
    return [
        app_commands.Choice(name=p, value=p)
        for p in PAIRS
        if current.upper() in p
    ][:25]  # Discord max 25 choices


async def autocomplete_timeframe(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Dropdown timeframe."""
    return [
        app_commands.Choice(name=tf.upper(), value=tf)
        for tf in TIMEFRAMES
        if current.lower() in tf.lower()
    ]


async def autocomplete_outcome(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Dropdown outcome untuk /outcome command."""
    options = ["TP1", "TP2", "SL1", "SL2", "MANUAL_CLOSE"]
    return [
        app_commands.Choice(name=o, value=o)
        for o in options
        if current.upper() in o
    ]

# ==================================================
# SLASH COMMANDS
# ==================================================

@client.tree.command(name="menu", description="Buka menu utama AI Trading Bot")
async def menu(interaction: discord.Interaction):
    embed = discord.Embed(
        title       = "🤖 AI Institutional Trading Bot",
        description = "Pilih aksi yang ingin kamu lakukan:",
        color       = discord.Color.gold(),
    )
    embed.add_field(name="🏆 Best Signal",        value="Sinyal terbaik dari semua pair",         inline=False)
    embed.add_field(name="📈 Signal per Pair",    value="Sinyal spesifik untuk satu pair",        inline=False)
    embed.add_field(name="📊 Chart per Pair",     value="Candle chart tanpa sinyal",              inline=False)
    embed.add_field(name="🔁 Scan Semua Pair",    value="Ringkasan cepat semua pair",             inline=False)
    embed.add_field(name="⚙️ Auto Signal Toggle", value="Aktifkan/matikan auto-signal otomatis",  inline=False)
    await interaction.response.send_message(embed=embed, view=MainMenuView())

@client.tree.command(name="signal", description="Best AI signal dari semua pair")
@app_commands.describe(timeframe="Pilih timeframe")
@app_commands.autocomplete(timeframe=autocomplete_timeframe)
async def signal_cmd(interaction: discord.Interaction, timeframe: str = DEFAULT_TF):
    if timeframe not in TIMEFRAMES:
        await interaction.response.send_message(
            f"❌ Timeframe tidak valid. Pilih: {', '.join(TIMEFRAMES.keys())}", ephemeral=True
        )
        return
    await interaction.response.defer()
    await interaction.followup.send(f"⏳ Scanning semua pair `{timeframe.upper()}`...")
    await scan_best_and_send(interaction, timeframe)


@client.tree.command(name="pair", description="Sinyal AI untuk pair tertentu")
@app_commands.describe(pair="Pilih pair", timeframe="Pilih timeframe")
@app_commands.autocomplete(pair=autocomplete_pair, timeframe=autocomplete_timeframe)
async def pair_cmd(interaction: discord.Interaction, pair: str, timeframe: str = DEFAULT_TF):
    pair = pair.upper()
    if pair not in PAIRS:
        await interaction.response.send_message(
            f"❌ Pair tidak tersedia.\nAvailable: {', '.join(PAIRS.keys())}", ephemeral=True
        )
        return
    if timeframe not in TIMEFRAMES:
        await interaction.response.send_message(
            f"❌ Timeframe tidak valid. Pilih: {', '.join(TIMEFRAMES.keys())}", ephemeral=True
        )
        return
    await interaction.response.defer()
    await send_signal_or_chart(interaction, pair, timeframe, mode="signal")


@client.tree.command(name="chart", description="Candle chart pair tertentu")
@app_commands.describe(pair="Pilih pair", timeframe="Pilih timeframe")
@app_commands.autocomplete(pair=autocomplete_pair, timeframe=autocomplete_timeframe)
async def chart_cmd(interaction: discord.Interaction, pair: str, timeframe: str = DEFAULT_TF):
    pair = pair.upper()
    if pair not in PAIRS:
        await interaction.response.send_message(
            f"❌ Pair tidak tersedia.\nAvailable: {', '.join(PAIRS.keys())}", ephemeral=True
        )
        return
    await interaction.response.defer()
    await send_signal_or_chart(interaction, pair, timeframe, mode="chart")


@client.tree.command(name="scanall", description="Scan semua pair sekaligus")
@app_commands.describe(timeframe="Pilih timeframe")
@app_commands.autocomplete(timeframe=autocomplete_timeframe)
async def scanall_cmd(interaction: discord.Interaction, timeframe: str = DEFAULT_TF):
    await interaction.response.defer()
    await interaction.followup.send(f"⏳ Scanning semua pair `{timeframe.upper()}`...")
    await scan_all_pairs_and_send(interaction, timeframe)


@client.tree.command(name="auto", description="Toggle auto signal (on/off)")
async def auto_cmd(interaction: discord.Interaction):
    global AUTO_SIGNAL
    AUTO_SIGNAL = not AUTO_SIGNAL
    status = "🟢 AKTIF" if AUTO_SIGNAL else "🔴 MATI"
    await interaction.response.send_message(f"Auto Signal: **{status}**")


@client.tree.command(name="divstatus", description="Lihat status RSI Divergence tracker")
async def divstatus_cmd(interaction: discord.Interaction):
    now   = time.time()
    lines = []
    for pair in PAIRS:
        for tf in ["15m", "1h", "4h"]:
            key  = f"{pair}_{tf}_div"
            last = _div_sent.get(key)
            if last:
                elapsed = int(now - last["ts"])
                remain  = max(0, DIV_COOLDOWN - elapsed)
                emoji   = "🔼" if last["type"] == "BULLISH_DIV" else "🔽"
                lines.append(
                    f"{emoji} **{pair}** `{tf}` — {last['type']} | "
                    f"dikirim {elapsed//60}m lalu | cooldown sisa {remain//60}m"
                )
    if not lines:
        lines = ["Belum ada RSI Divergence yang dikirim sejak bot start."]

    embed = discord.Embed(
        title       = "📊 RSI Divergence Tracker Status",
        description = "\n".join(lines),
        color       = discord.Color.blurple(),
    )
    embed.set_footer(text=f"Cooldown: {DIV_COOLDOWN//60} menit per pair | Scan setiap 5 menit")
    await interaction.response.send_message(embed=embed, ephemeral=True)


@client.tree.command(name="datasource", description="Cek status data source")
async def datasource_cmd(interaction: discord.Interaction):
    _check_tv_available()
    source     = "🟢 **TradingView WebSocket**" if _tv_ok else "🟡 **Yahoo Finance** (fallback)"
    cache_info = f"Cache entries: `{len(_data_cache)}` | TTL: `{CACHE_TTL}s`"
    await interaction.response.send_message(
        f"**Data Source:** {source}\n{cache_info}", ephemeral=True
    )


@client.tree.command(name="dbstats", description="Statistik database & ML model")
async def dbstats_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        stats    = get_db_stats()
        win_rate = get_win_rate()
        ml_stat  = ml_status()

        embed = discord.Embed(title="📊 Database & ML Stats", color=discord.Color.blurple())

        if stats:
            embed.add_field(
                name  = "🗃 Database",
                value = (
                    f"Signals logged: `{stats.get('signals', 0)}`\n"
                    f"Div alerts:     `{stats.get('div_alerts', 0)}`\n"
                    f"Score history:  `{stats.get('score_history', 0)}`\n"
                    f"Outcomes:       `{stats.get('outcomes', 0)}`"
                ),
                inline=True,
            )
        else:
            embed.add_field(name="🗃 Database", value="❌ Gagal mengambil statistik", inline=True)

        if win_rate and win_rate.get("total", 0) > 0:
            embed.add_field(
                name  = "🏆 Win Rate",
                value = (
                    f"Total: `{win_rate['total']}`\n"
                    f"Wins:  `{win_rate['wins']}`\n"
                    f"Loss:  `{win_rate['losses']}`\n"
                    f"Rate:  `{win_rate['win_rate']}%`"
                ),
                inline=True,
            )
        else:
            embed.add_field(name="🏆 Win Rate", value="⏳ Belum ada data outcome", inline=True)

        embed.add_field(name="🤖 ML Model", value=ml_stat or "⏳ Belum siap", inline=False)
        await interaction.followup.send(embed=embed, ephemeral=True)

    except Exception as e:
        print(f"[DBSTATS ERROR] {e}")
        await interaction.followup.send(f"❌ Error: `{e}`", ephemeral=True)


async def autocomplete_action(
    interaction: discord.Interaction,
    current: str,
) -> list[app_commands.Choice[str]]:
    """Dropdown action BUY / SELL."""
    options = ["BUY", "SELL"]
    return [
        app_commands.Choice(name=o, value=o)
        for o in options
        if current.upper() in o
    ]
 
 
@client.tree.command(name="outcome", description="Catat hasil trade (TP/SL hit)")
@app_commands.describe(
    pair       = "Pilih pair",
    timeframe  = "Pilih timeframe",
    action     = "Pilih arah trade",
    outcome    = "Pilih hasil trade",
    entry      = "Harga entry",
    exit_price = "Harga exit aktual",
)
@app_commands.autocomplete(
    pair      = autocomplete_pair,
    timeframe = autocomplete_timeframe,
    action    = autocomplete_action,
    outcome   = autocomplete_outcome,
)
async def outcome_cmd(
    interaction: discord.Interaction,
    pair:        str,
    timeframe:   str,
    action:      str,           # ← sekarang dari input user, bukan dihitung
    outcome:     str,
    entry:       float,
    exit_price:  float,
):
    pair    = pair.upper()
    action  = action.upper()
    outcome = outcome.upper()
 
    # validasi
    if action not in ("BUY", "SELL"):
        await interaction.response.send_message(
            "❌ Action tidak valid. Pilih: BUY / SELL", ephemeral=True
        )
        return
 
    valid_outcomes = {"TP1", "TP2", "SL1", "SL2", "MANUAL_CLOSE"}
    if outcome not in valid_outcomes:
        await interaction.response.send_message(
            f"❌ Outcome tidak valid. Pilih: {', '.join(valid_outcomes)}", ephemeral=True
        )
        return
 
    # hitung pips & RR berdasarkan action user, bukan asumsi
    if action == "BUY":
        pips_result = exit_price - entry
    else:  # SELL
        pips_result = entry - exit_price
 
    recent      = get_recent_signals(pair, limit=5)
    rr_achieved = 0.0
    if recent:
        last    = recent[0]
        sl_used = last.get("sl2")
        if sl_used and abs(entry - sl_used) > 0:
            rr_achieved = abs(exit_price - entry) / abs(entry - sl_used)
 
    log_outcome(
        signal_id   = recent[0]["id"] if recent else 0,
        pair        = pair,
        timeframe   = timeframe,
        action      = action,           # ← dari input user
        entry       = entry,
        tp1         = recent[0].get("tp1", 0) if recent else 0,
        tp2         = recent[0].get("tp2", 0) if recent else 0,
        sl1         = recent[0].get("sl1", 0) if recent else 0,
        sl2         = recent[0].get("sl2", 0) if recent else 0,
        outcome     = outcome,
        pips_result = pips_result,
        rr_achieved = round(rr_achieved, 2),
    )
 
    # trigger background retrain
    def _bg_train():
        rows = get_training_data()
        if len(rows) >= int(os.getenv("ML_MIN_ROWS", "10")):
            train(rows)
    threading.Thread(target=_bg_train, daemon=True).start()
 
    win   = outcome in ("TP1", "TP2")
    emoji = "✅" if win else "❌"
    color = discord.Color.green() if win else discord.Color.red()
 
    embed = discord.Embed(
        title = f"{emoji} Outcome Dicatat — {pair} {timeframe.upper()}",
        color = color,
    )
    embed.add_field(
        name  = "📋 Detail Trade",
        value = (
            f"Action:    `{action}`\n"
            f"Entry:     `{entry:.4f}`\n"
            f"Exit:      `{exit_price:.4f}`\n"
            f"Outcome:   `{outcome}`"
        ),
        inline=True,
    )
    embed.add_field(
        name  = "📊 Hasil",
        value = (
            f"PnL:  `{pips_result:+.4f}`\n"
            f"RR:   `1:{rr_achieved:.2f}`\n"
            f"{'🏆 Win — data ML diupdate' if win else '📉 Loss — data ML diupdate'}"
        ),
        inline=True,
    )
    embed.set_footer(
        text=f"ML retrain otomatis setelah {os.getenv('ML_MIN_ROWS','10')} outcomes"
    )
 
    await interaction.response.send_message(embed=embed, ephemeral=True)
 


@client.tree.command(name="regime", description="Cek market regime pair saat ini")
@app_commands.describe(pair="Pilih pair", timeframe="Pilih timeframe")
@app_commands.autocomplete(pair=autocomplete_pair, timeframe=autocomplete_timeframe)
async def regime_cmd(interaction: discord.Interaction, pair: str, timeframe: str = DEFAULT_TF):
    pair = pair.upper()
    if pair not in PAIRS:
        await interaction.response.send_message("❌ Pair tidak valid", ephemeral=True)
        return

    await interaction.response.defer()
    try:
        df     = await run_blocking(get_data, pair, timeframe)
        regime = detect_regime(df)

        color_map = {
            "TREND_UP":   discord.Color.green(),
            "TREND_DOWN": discord.Color.red(),
            "BREAKOUT":   discord.Color.orange(),
            "REVERSAL":   discord.Color.purple(),
            "RANGING":    discord.Color.greyple(),
            "UNKNOWN":    discord.Color.default(),
        }
        embed = discord.Embed(
            title = f"{regime['emoji']} Market Regime — {pair} {timeframe.upper()}",
            color = color_map.get(regime["regime"], discord.Color.default()),
        )
        embed.add_field(name="📊 Regime Detail", value=regime_embed_value(regime), inline=False)
        embed.add_field(
            name  = "🔑 Alasan",
            value = "\n".join(f"• {r}" for r in regime["reasons"]) or "N/A",
            inline=False,
        )
        embed.add_field(
            name  = "📐 Modifier",
            value = (
                f"BUY signal:  `{regime_score_modifier(regime, 'BUY'):+d}` pts\n"
                f"SELL signal: `{regime_score_modifier(regime, 'SELL'):+d}` pts"
            ),
            inline=True,
        )
        await interaction.followup.send(embed=embed)
    except Exception as e:
        await interaction.followup.send(f"❌ Error: `{e}`")


# ==================================================
# RSI DIVERGENCE NOTIFICATION SYSTEM
# ==================================================

_div_sent: dict = {}
DIV_COOLDOWN = 3600


def _rsi_div_detail(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 5) -> dict:
    close      = df["Close"]
    price_now  = float(close.iloc[-1])
    price_prev = float(close.iloc[-lookback])
    rsi_now    = float(rsi_series.iloc[-1])
    rsi_prev   = float(rsi_series.iloc[-lookback])

    price_diff = price_now  - price_prev
    rsi_diff   = rsi_now    - rsi_prev

    if price_diff < 0 and rsi_diff > 0:
        div_type = "BULLISH_DIV"
        strength = abs(rsi_diff)
    elif price_diff > 0 and rsi_diff < 0:
        div_type = "BEARISH_DIV"
        strength = abs(rsi_diff)
    else:
        div_type = "NONE"
        strength = 0.0

    return {
        "type":       div_type,
        "strength":   strength,
        "price_now":  price_now,
        "price_prev": price_prev,
        "rsi_now":    rsi_now,
        "rsi_prev":   rsi_prev,
        "price_diff": price_diff,
        "rsi_diff":   rsi_diff,
        "lookback":   lookback,
    }


def build_div_embed(pair: str, timeframe: str, div: dict, signal_data: dict) -> discord.Embed:
    is_bull  = div["type"] == "BULLISH_DIV"
    color    = discord.Color.green() if is_bull else discord.Color.red()
    emoji    = "🔼" if is_bull else "🔽"
    div_name = "BULLISH" if is_bull else "BEARISH"

    strength = div["strength"]
    if strength >= 15:   str_lbl = "STRONG"
    elif strength >= 8:  str_lbl = "MODERATE"
    else:                str_lbl = "WEAK"

    bos    = div.get("bos", {})
    candle = div.get("candle", {})

    bos_dir    = "Bull BOS" if bos.get("fresh_bull") else "Bear BOS"
    bos_level  = bos.get("swing_high" if is_bull else "swing_low", 0)
    candle_ok  = candle.get("confirmed", False)
    candle_typ = candle.get("candle_type", "-")
    body_pct   = candle.get("body_pct", 0)
    vol_ok     = candle.get("vol_ok", False)

    embed = discord.Embed(
        title = f"{emoji} RSI {div_name} DIVERGENCE — {pair}  [{timeframe.upper()}]",
        description = (
            f"**Kekuatan:** `{str_lbl}` ({strength:.1f} RSI pts)  |  "
            f"**AI Score:** `{signal_data['score']:.1f}%` — {signal_data['confidence']}"
        ),
        color = color,
    )

    arrow_price = "↓" if div["price_diff"] < 0 else "↑"
    arrow_rsi   = "↑" if div["rsi_diff"]   > 0 else "↓"

    if is_bull:
        explanation = (
            "Harga cetak **Lower Low** tapi RSI cetak **Higher Low**\n"
            "→ Momentum bearish melemah, potensi reversal naik\n"
            "→ Smart money kemungkinan **akumulasi** di area ini"
        )
    else:
        explanation = (
            "Harga cetak **Higher High** tapi RSI cetak **Lower High**\n"
            "→ Momentum bullish melemah, potensi reversal turun\n"
            "→ Smart money kemungkinan **distribusi** di area ini"
        )

    embed.add_field(name="📊 Analisis Divergence", value=explanation, inline=False)
    embed.add_field(
        name  = "📈 Data RSI",
        value = (
            f"Harga  {div['lookback']}c lalu → sekarang: "
            f"`{div['price_prev']:.4f}` {arrow_price} `{div['price_now']:.4f}`\n"
            f"RSI    {div['lookback']}c lalu → sekarang: "
            f"`{div['rsi_prev']:.2f}` {arrow_rsi} `{div['rsi_now']:.2f}`"
        ),
        inline=False,
    )

    bos_status = "FRESH" if (bos.get("fresh_bull") if is_bull else bos.get("fresh_bear")) else "NOT CONFIRMED"
    embed.add_field(
        name  = "🏗 BOS Konfirmasi",
        value = (
            f"Status: `{bos_dir}` — `{bos_status}`\n"
            f"Level tembus: `{bos_level:.4f}`\n"
            f"{'Break di atas swing high → struktur bullish terkonfirmasi' if is_bull else 'Break di bawah swing low → struktur bearish terkonfirmasi'}"
        ),
        inline=True,
    )

    candle_status = "CONFIRMED" if candle_ok else ("DISABLED" if not candle else "NOT CONFIRMED")
    embed.add_field(
        name  = "🕯 Candle Konfirmasi",
        value = (
            "⚡ Dinonaktifkan — alert tanpa tunggu candle"
            if not candle
            else (
                f"Status: `{candle_status}`\n"
                f"Tipe: `{candle_typ}`  Body: `{body_pct*100:.0f}%`\n"
                f"Volume: `{'Above avg' if vol_ok else 'Below avg'}`\n"
                f"{'Close > mid candle sebelumnya' if is_bull else 'Close < mid candle sebelumnya'}"
            )
        ),
        inline=True,
    )

    sw = signal_data["sweep"]
    embed.add_field(
        name  = "📍 Level Trading",
        value = (
            f"Entry: `{signal_data['entry']:.4f}`\n"
            f"TP1: `{signal_data['tp1']:.4f}` (RR `1:{signal_data['rr1']:.2f}`)\n"
            f"TP2: `{signal_data['tp2']:.4f}` (RR `1:{signal_data['rr2']:.2f}`)\n"
            f"SL1: `{signal_data['sl1']:.4f}` tight\n"
            f"SL2: `{signal_data['sl2']:.4f}` anti-sweep\n"
            f"Pool ref: `{sw['sweep_ref']:.4f}`"
        ),
        inline=True,
    )

    embed.add_field(
        name  = "🧠 Konfirmasi Lain",
        value = (
            f"MTF Bias: `{signal_data['mtf']['bias']}`\n"
            f"Orderflow: `{signal_data['orderflow']['bias']}`\n"
            f"FVG: `{'IN BULL FVG' if signal_data['fvg']['in_bull_fvg'] else 'IN BEAR FVG' if signal_data['fvg']['in_bear_fvg'] else 'Outside'}`\n"
            f"Session: `{signal_data['session']['session']}` {signal_data['session']['utc_time']}\n"
            f"Regime: `{signal_data.get('regime', {}).get('regime', 'N/A')}`"
        ),
        inline=True,
    )

    embed.add_field(
        name  = "⚠️ Disclaimer",
        value = "RSI Divergence adalah sinyal probabilistik. Selalu gunakan manajemen risiko.",
        inline=False,
    )
    embed.set_footer(text=f"RSI Div Alert | TF: {timeframe.upper()} | Filter: BOS+Candle+Score>=65 | AI Engine v4")
    return embed


def create_div_chart(df, pair, timeframe, div, signal_data) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    df_c = df.copy()
    if not isinstance(df_c.index, pd.DatetimeIndex):
        df_c.index = pd.to_datetime(df_c.index)

    candle_limit = {"5m": 80, "15m": 80, "1h": 72, "4h": 48}
    n    = candle_limit.get(timeframe, 80)
    df_c = df_c.iloc[-n:]

    close  = df_c["Close"]
    rsi_s  = rsi(close)

    df_c["EMA8"]  = ema(close, 8)
    df_c["EMA21"] = ema(close, 21)
    df_c["EMA50"] = ema(close, 50)

    xs     = np.arange(len(df_c))
    opens  = df_c["Open"].values
    highs  = df_c["High"].values
    lows   = df_c["Low"].values
    closes = df_c["Close"].values
    rsi_v  = rsi_s.values

    is_bull  = div["type"] == "BULLISH_DIV"
    is_buy   = signal_data["action"] == "BUY"
    entry    = signal_data["entry"]
    tp1      = signal_data["tp1"]
    tp2      = signal_data["tp2"]
    sl1      = signal_data["sl1"]
    sl2      = signal_data["sl2"]

    fig = plt.figure(figsize=(18, 11), facecolor="#0d1117")
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[3, 1.2],
                   hspace=0.04, left=0.01, right=0.85, top=0.93, bottom=0.07)
    ax_c = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1])
    ax_c.set_facecolor("#0d1117")
    ax_r.set_facecolor("#0d1117")

    if is_buy:
        ax_c.axhspan(entry, tp2, alpha=0.07, color="#00c853", zorder=1)
        ax_c.axhspan(entry, tp1, alpha=0.13, color="#69f0ae", zorder=1)
        ax_c.axhspan(sl2, entry, alpha=0.08, color="#d50000", zorder=1)
        if sl1: ax_c.axhspan(sl1, entry, alpha=0.13, color="#ff6d00", zorder=1)
    else:
        ax_c.axhspan(tp2, entry, alpha=0.07, color="#00c853", zorder=1)
        ax_c.axhspan(tp1, entry, alpha=0.13, color="#69f0ae", zorder=1)
        ax_c.axhspan(entry, sl2, alpha=0.08, color="#d50000", zorder=1)
        if sl1: ax_c.axhspan(entry, sl1, alpha=0.13, color="#ff6d00", zorder=1)

    w = 0.6
    for i, x in enumerate(xs):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        col = "#26a69a" if c >= o else "#ef5350"
        ax_c.plot([x, x], [l, h], color=col, linewidth=0.8, zorder=6)
        body_lo = min(o, c)
        body_hi = max(o, c)
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, body_lo), w,
            max(body_hi - body_lo, (h - l) * 0.003),
            boxstyle="square,pad=0", linewidth=0, facecolor=col, zorder=7,
        )
        ax_c.add_patch(rect)

    ax_c.plot(xs, df_c["EMA8"].values,  color="#00e5ff", lw=0.9, label="EMA 8",  zorder=8)
    ax_c.plot(xs, df_c["EMA21"].values, color="#ffeb3b", lw=0.9, label="EMA 21", zorder=8)
    ax_c.plot(xs, df_c["EMA50"].values, color="#ff9800", lw=1.1, label="EMA 50", zorder=8)

    lx = xs[-1] + 0.5
    ax_c.axhline(tp2,   color="#00e676", lw=1.5, linestyle="--", zorder=9)
    ax_c.text(lx, tp2,  f" TP2 {tp2:.4f}", color="#00e676",  fontsize=7.5, va="center", fontweight="bold")
    ax_c.axhline(tp1,   color="#b9f6ca", lw=1.2, linestyle="-.", zorder=9)
    ax_c.text(lx, tp1,  f" TP1 {tp1:.4f}", color="#b9f6ca",  fontsize=7.5, va="center", fontweight="bold")
    ax_c.axhline(entry, color="#ffffff", lw=1.0, linestyle="--", zorder=9, alpha=0.85)
    ax_c.text(lx, entry,f" ENTRY {entry:.4f}", color="#ffffff", fontsize=7.5, va="center", fontweight="bold")
    ax_c.axhline(sl2,   color="#ff1744", lw=1.5, linestyle="--", zorder=9)
    ax_c.text(lx, sl2,  f" SL2 {sl2:.4f}", color="#ff1744",  fontsize=7.5, va="center", fontweight="bold")
    if sl1:
        ax_c.axhline(sl1, color="#ffab40", lw=1.2, linestyle="-.", zorder=9)
        ax_c.text(lx, sl1, f" SL1 {sl1:.4f}", color="#ffab40", fontsize=7.5, va="center", fontweight="bold")

    lookback  = div["lookback"]
    x_prev    = xs[-lookback]
    x_now     = xs[-1]
    y_prev    = div["price_prev"]
    y_now     = div["price_now"]
    div_color = "#00e676" if is_bull else "#ff1744"

    ax_c.scatter([x_prev, x_now], [y_prev, y_now], color=div_color, s=60, zorder=10, edgecolors="white", linewidths=0.5)
    ax_c.annotate("", xy=(x_now, y_now), xytext=(x_prev, y_prev),
                  arrowprops=dict(arrowstyle="->", color=div_color, lw=1.5, connectionstyle="arc3,rad=0.1"))
    mid_x = (x_prev + x_now) / 2
    mid_y = (y_prev + y_now) / 2
    ax_c.text(mid_x, mid_y, f" {'BULL DIV' if is_bull else 'BEAR DIV'}",
              color=div_color, fontsize=9, fontweight="bold",
              bbox=dict(facecolor="#0d1117", edgecolor=div_color, boxstyle="round,pad=0.3", alpha=0.85))

    all_levels = [min(lows), max(highs), tp2, sl2]
    if sl1: all_levels.append(sl1)
    y_lo = min(all_levels); y_hi = max(all_levels)
    pad  = (y_hi - y_lo) * 0.07
    ax_c.set_ylim(y_lo - pad, y_hi + pad)
    ax_c.set_xlim(-1, xs[-1] + 12)
    ax_c.set_xticks([])
    ax_c.yaxis.tick_right()
    ax_c.yaxis.set_tick_params(labelcolor="#aaaaaa", labelsize=7.5)
    ax_c.grid(axis="y", color="#1f2937", lw=0.4)
    for sp in ax_c.spines.values(): sp.set_edgecolor("#1f2937")
    ax_c.legend(loc="upper left", fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#cccccc", ncol=3, framealpha=0.85)

    valid = ~np.isnan(rsi_v)
    ax_r.plot(xs[valid], rsi_v[valid], color="#ce93d8", lw=1.2, label="RSI 14", zorder=5)
    ax_r.axhspan(70, 100, alpha=0.08, color="#ff1744")
    ax_r.axhspan(0,   30, alpha=0.08, color="#00e676")
    ax_r.axhline(70, color="#ff1744", lw=0.6, linestyle=":", alpha=0.7)
    ax_r.axhline(50, color="#ffffff", lw=0.5, linestyle=":", alpha=0.3)
    ax_r.axhline(30, color="#00e676", lw=0.6, linestyle=":", alpha=0.7)
    ax_r.fill_between(xs[valid], rsi_v[valid], 50, where=rsi_v[valid] > 50, alpha=0.08, color="#00e676")
    ax_r.fill_between(xs[valid], rsi_v[valid], 50, where=rsi_v[valid] < 50, alpha=0.08, color="#ff1744")

    rsi_prev_val = div["rsi_prev"]
    rsi_now_val  = div["rsi_now"]
    ax_r.scatter([x_prev, x_now], [rsi_prev_val, rsi_now_val], color=div_color, s=60, zorder=10, edgecolors="white", linewidths=0.5)
    ax_r.annotate("", xy=(x_now, rsi_now_val), xytext=(x_prev, rsi_prev_val),
                  arrowprops=dict(arrowstyle="->", color=div_color, lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax_r.text(x_now + 0.3, rsi_now_val, f" RSI {rsi_now_val:.1f}", color="#ce93d8", fontsize=7.5, va="center")

    ax_r.set_xlim(-1, xs[-1] + 12)
    ax_r.set_ylim(0, 100)
    ax_r.yaxis.tick_right()
    ax_r.yaxis.set_tick_params(labelcolor="#aaaaaa", labelsize=7)
    ax_r.set_ylabel("RSI", color="#aaaaaa", fontsize=8)
    ax_r.yaxis.set_label_position("left")

    tick_every  = max(1, len(xs) // 10)
    tick_idx    = xs[::tick_every]
    tick_labels = [df_c.index[i].strftime("%m/%d %H:%M") for i in tick_idx]
    ax_r.set_xticks(tick_idx)
    ax_r.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=6.5, color="#aaaaaa")
    ax_r.grid(axis="y", color="#1f2937", lw=0.4)
    for sp in ax_r.spines.values(): sp.set_edgecolor("#1f2937")

    div_lbl = "BULLISH DIV 🔼" if is_bull else "BEARISH DIV 🔽"
    fig.suptitle(
        f"RSI {div_lbl}  ·  {pair}  ·  {timeframe.upper()}  ·  Score {signal_data['score']:.1f}%",
        color=div_color, fontsize=13, fontweight="bold", y=0.97,
    )
    ax_c.text(0.5, 0.5, "APC-Institional Engine by Yor", transform=ax_c.transAxes,
              fontsize=28, color="white", alpha=0.03, ha="center", va="center", rotation=30, fontweight="bold")

    filename = f"{pair}_div_chart.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename


async def check_and_send_div(channel, pair: str, timeframe: str = DEFAULT_TF):
    global _div_sent

    # ── Filter 1: hanya 15m ke atas ───────────────
    if timeframe not in ("15m", "1h", "4h"):
        return

    try:
        data     = await run_blocking(generate_signal, pair, timeframe)
        div_type = data["rsi_div"]
        df       = data["df"]
        close    = df["Close"]

        # ── Filter 2: divergence harus ada ────────
        if div_type == "NONE":
            return

        # ── Filter 3: BOS konfirmasi ───────────────
        bos = bos_confirmation(df, swing=10)
        if div_type == "BULLISH_DIV":
            if not bos["fresh_bull"]:
                print(f"[DIV SKIP] {pair} {timeframe}: BULLISH_DIV tapi no fresh bull BOS")
                return
        else:
            if not bos["fresh_bear"]:
                print(f"[DIV SKIP] {pair} {timeframe}: BEARISH_DIV tapi no fresh bear BOS")
                return

        # Filter 4 (candle konfirmasi) DINONAKTIFKAN
        # Alert dikirim langsung tanpa menunggu candle konfirmasi

        # ── Filter 5: AI score ────────────────────
        if data["score"] < 65:
            print(f"[DIV SKIP] {pair} {timeframe}: score {data['score']} < 65")
            return

        # ── Filter 6: cooldown ────────────────────
        cache_key = f"{pair}_{timeframe}_div"
        now       = time.time()
        last      = _div_sent.get(cache_key, {})
        if last.get("type") == div_type and (now - last.get("ts", 0)) < DIV_COOLDOWN:
            return

        # ── Semua filter lolos → kirim ────────────
        rsi_s   = rsi(close)
        div_det = _rsi_div_detail(df, rsi_s, lookback=5)

        div_det["bos"]    = bos
        div_det["candle"] = {}   # candle konfirmasi dinonaktifkan

        embed = build_div_embed(pair, timeframe, div_det, data)
        path  = await run_blocking(create_div_chart, df, pair, timeframe, div_det, data)

        await channel.send(
            content = (
                f"🔔 **RSI DIV ALERT** — `{pair}` `{timeframe.upper()}`\n"
                f"{'🔼 BULLISH' if div_type == 'BULLISH_DIV' else '🔽 BEARISH'} | "
                f"BOS: {'Fresh Bull' if bos['fresh_bull'] else 'Fresh Bear'} | "
                f"Score: `{data['score']:.1f}%` | {data['confidence']}"
            ),
            embed = embed,
            file  = discord.File(path),
        )

        # Log to DB
        try:
            log_div_alert(pair, timeframe, div_det, data)
        except Exception as e:
            print(f"[DB log_div ERROR] {e}")

        _div_sent[cache_key] = {"type": div_type, "ts": now}
        print(f"[DIV ALERT SENT] {pair} {timeframe} {div_type} score={data['score']:.1f}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[DIV CHECK ERROR] {pair} {timeframe}:\n{tb}")


async def div_scan_loop():
    await client.wait_until_ready()
    await asyncio.sleep(180)

    channel = client.get_channel(CHANNEL_ID)
    div_tfs = ["15m", "1h", "4h"]

    while not client.is_closed():
        try:
            if channel:
                for pair in PAIRS:
                    for tf in div_tfs:
                        await check_and_send_div(channel, pair, tf)
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"[DIV LOOP ERROR] {e}")
        await asyncio.sleep(300)


async def auto_signal_loop():
    await client.wait_until_ready()
    await asyncio.sleep(30)

    channel = client.get_channel(CHANNEL_ID)

    while not client.is_closed():
        try:
            if AUTO_SIGNAL and channel:
                for pair in PAIRS:
                    try:
                        data = await run_blocking(generate_signal, pair, DEFAULT_TF)
                        if data["score"] >= AUTO_MIN_SCORE:
                            path = await run_blocking(
                                create_chart, data["df"], pair, data["entry"],
                                data["tp"], data["sl"], data["timeframe"],
                                data["sweep"], data["fvg"],
                                data["tp1"], data["sl1"], data["rr1"], data["rr2"]
                            )
                            embed = build_embed(data)
                            view  = SignalActionView(data)
                            await channel.send(embed=embed, view=view, file=discord.File(path))
                    except Exception as e:
                        print(f"AUTO ERROR {pair}: {e}")
                    await asyncio.sleep(2)
            await asyncio.sleep(900)
        except Exception as e:
            print(f"LOOP ERROR: {e}")
            await asyncio.sleep(30)


# ==================================================
# STARTUP HELPER
# ==================================================
def _try_load_and_train():
    """
    Saat startup:
    1. Coba load model dari disk (ada jika tidak di Railway)
    2. Jika tidak ada → langsung retrain dari data Supabase/SQLite
    3. Jika data kurang → log info, bot tetap jalan dengan rule-based
    """
    loaded = load_model()
 
    if not loaded:
        print("[ML] Mencoba retrain dari database...")
        try:
            rows = get_training_data()
            if not rows:
                print(f"[ML] Belum ada data outcomes di DB — butuh minimal {ML_MIN_ROWS} rows")
                print("[ML] Bot tetap jalan dengan rule-based scoring")
                return
 
            print(f"[ML] Ditemukan {len(rows)} rows di DB → mulai training...")
            success = train(rows)
            if success:
                print("[ML] ✅ Model berhasil ditraining dari data DB")
            else:
                print(f"[ML] ⚠️  Training gagal — butuh minimal {ML_MIN_ROWS} outcomes dengan label TP/SL")
                print("[ML] Bot tetap jalan dengan rule-based scoring")
        except Exception as e:
            print(f"[ML] Error saat retrain startup: {e}")
            print("[ML] Bot tetap jalan dengan rule-based scoring")

# ==================================================
# START
# ==================================================
@client.event
async def on_ready():
    try:
        await client.tree.sync()
        print("✅ Slash commands synced")
    except Exception as e:
        print(f"⚠️  Slash sync error: {e}")
 
    _check_tv_available()
    init_db()               # koneksi Supabase/SQLite
    _try_load_and_train()   # load model atau retrain dari DB
 
    print(f"✅ AI TRADING BOT READY — {client.user}")
    print(f"   Data source: {'TradingView WS' if _tv_ok else 'Yahoo Finance'}")
    print(f"   DB: {'PostgreSQL' if os.getenv('DATABASE_URL') else 'SQLite'}")

async def main():
    async with client:
        client.loop.create_task(auto_signal_loop())
        client.loop.create_task(div_scan_loop())
        await client.start(TOKEN)

asyncio.run(main())
