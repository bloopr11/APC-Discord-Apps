# ==================================================
# regime.py  —  Market Regime Detection
#
# Mendeteksi 5 regime pasar:
#   TREND_UP    — trending naik kuat (ADX > 25, +DI > -DI)
#   TREND_DOWN  — trending turun kuat
#   RANGING     — sideways/konsolidasi (ADX < 20)
#   BREAKOUT    — harga baru saja keluar dari range (volatilitas melonjak)
#   REVERSAL    — tanda pembalikan arah (struktur + momentum)
#
# Regime menentukan:
#   - Strategy yang tepat (trend-follow vs mean-revert)
#   - Score modifier akhir (+/- hingga 15 poin)
#   - Filter entry: beberapa setup di-skip di regime tertentu
# ==================================================

import numpy as np
import pandas as pd
from typing import Tuple

# ==================================================
# INTERNAL HELPERS  (tidak import dari bot utama
# agar regime.py bisa berdiri sendiri)
# ==================================================
def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def _adx_full(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI) series."""
    high  = df["High"]
    low   = df["Low"]
    atr_s = _atr(df, period)

    up_move   = high.diff()
    down_move = -low.diff()

    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_di  = pd.Series(plus_dm,  index=df.index).rolling(period).mean() / atr_s.replace(0, np.nan) * 100
    minus_di = pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr_s.replace(0, np.nan) * 100

    dx  = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _bb_width(series: pd.Series, period: int = 20, std: float = 2.0) -> pd.Series:
    """Bollinger Band Width sebagai % dari SMA."""
    sma = series.rolling(period).mean()
    s   = series.rolling(period).std()
    return (2 * std * s) / sma.replace(0, np.nan) * 100

# ==================================================
# REGIME CONSTANTS
# ==================================================
REGIME_LABELS = {
    "TREND_UP":   {"emoji": "📈", "color": "green",  "strategy": "trend-follow"},
    "TREND_DOWN": {"emoji": "📉", "color": "red",    "strategy": "trend-follow"},
    "RANGING":    {"emoji": "↔️",  "color": "grey",   "strategy": "mean-revert"},
    "BREAKOUT":   {"emoji": "💥", "color": "orange", "strategy": "momentum"},
    "REVERSAL":   {"emoji": "🔄", "color": "purple", "strategy": "reversal"},
}

# Score modifier per regime × action
# (regime, action) → modifier
REGIME_SCORE_MOD = {
    ("TREND_UP",   "BUY"):  +12,
    ("TREND_UP",   "SELL"): -12,   # contra trend → penalized
    ("TREND_DOWN", "BUY"):  -12,
    ("TREND_DOWN", "SELL"): +12,
    ("RANGING",    "BUY"):   -3,   # slight penalty, bisa mean-revert
    ("RANGING",    "SELL"):  -3,
    ("BREAKOUT",   "BUY"):   +8,
    ("BREAKOUT",   "SELL"):  +8,   # direction agnostic — both can work
    ("REVERSAL",   "BUY"):   +6,
    ("REVERSAL",   "SELL"):  +6,
}

# Entry yang sebaiknya di-SKIP per regime
REGIME_SKIP_RULES = {
    "RANGING":    lambda action, score: score < 60,    # skip low-score entries di ranging
    "TREND_UP":   lambda action, score: action == "SELL" and score < 75,
    "TREND_DOWN": lambda action, score: action == "BUY"  and score < 75,
}

# ==================================================
# MAIN REGIME DETECTOR
# ==================================================
def detect_regime(df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Deteksi market regime dari OHLCV DataFrame.

    Returns dict dengan:
        regime      : str   — salah satu REGIME_LABELS
        confidence  : float — 0.0-1.0
        adx         : float
        plus_di     : float
        minus_di    : float
        bb_width    : float — current BB width %
        bb_squeeze  : bool  — BB width < 20th percentile → konsolidasi
        volatility  : str   — LOW | NORMAL | HIGH | EXTREME
        score_mod   : int   — modifier untuk advanced_score
        strategy    : str   — strategy yang cocok
        detail      : dict  — semua sub-indikator
    """
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    atr_s  = _atr(df)

    # ── ADX + DI ──────────────────────────────────
    adx_s, pdi_s, mdi_s = _adx_full(df)
    adx_val = float(adx_s.iloc[-1]) if not np.isnan(adx_s.iloc[-1]) else 20.0
    pdi_val = float(pdi_s.iloc[-1]) if not np.isnan(pdi_s.iloc[-1]) else 25.0
    mdi_val = float(mdi_s.iloc[-1]) if not np.isnan(mdi_s.iloc[-1]) else 25.0

    # ADX slope (rising = trend strengthening)
    adx_tail  = adx_s.iloc[-lookback:].dropna()
    adx_slope = float(np.polyfit(range(len(adx_tail)), adx_tail.values, 1)[0]) if len(adx_tail) > 1 else 0.0

    # ── BB Width + Squeeze ────────────────────────
    bbw_s     = _bb_width(close)
    bbw_now   = float(bbw_s.iloc[-1]) if not np.isnan(bbw_s.iloc[-1]) else 5.0
    bbw_pct20 = float(bbw_s.quantile(0.20)) if len(bbw_s.dropna()) > 20 else bbw_now
    bb_squeeze = bbw_now < bbw_pct20 * 1.1   # current ≤ 20th pct → squeeze

    # ATR ratio vs lookback mean (volatility measure)
    atr_now   = float(atr_s.iloc[-1])
    atr_mean  = float(atr_s.iloc[-20:].mean()) if len(atr_s) >= 20 else atr_now
    atr_ratio = atr_now / atr_mean if atr_mean > 0 else 1.0

    if   atr_ratio > 2.0:  volatility = "EXTREME"
    elif atr_ratio > 1.5:  volatility = "HIGH"
    elif atr_ratio < 0.7:  volatility = "LOW"
    else:                   volatility = "NORMAL"

    # ── RSI ───────────────────────────────────────
    rsi_val = float(_rsi(close).iloc[-1])

    # ── EMA slope ────────────────────────────────
    ema50_s   = _ema(close, 50)
    ema50_tail = ema50_s.iloc[-lookback:].dropna()
    ema_slope = float(np.polyfit(range(len(ema50_tail)), ema50_tail.values, 1)[0]) if len(ema50_tail) > 1 else 0.0

    # ── Price vs recent highs/lows ────────────────
    n_bars = min(20, len(df))
    recent_high = float(high.iloc[-n_bars:].max())
    recent_low  = float(low.iloc[-n_bars:].min())
    price       = float(close.iloc[-1])
    price_pos   = (price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5

    # ── SWING STRUCTURE (CHoCH check) ─────────────
    swing_high_prev = float(high.rolling(10).max().iloc[-2]) if len(df) > 10 else recent_high
    swing_low_prev  = float(low.rolling(10).min().iloc[-2])  if len(df) > 10 else recent_low
    bull_choch = price > swing_high_prev and float(close.iloc[-2]) <= swing_high_prev
    bear_choch = price < swing_low_prev  and float(close.iloc[-2]) >= swing_low_prev

    # ==================================================
    # REGIME CLASSIFICATION LOGIC
    # ==================================================
    regime     = "RANGING"
    confidence = 0.5
    reasons    = []

    # ── BREAKOUT: volatility lonjak + keluar dari squeeze
    if atr_ratio > 1.6 and not bb_squeeze and volatility in ("HIGH", "EXTREME"):
        regime     = "BREAKOUT"
        confidence = min(0.5 + (atr_ratio - 1.6) * 0.3, 0.95)
        reasons.append(f"ATR ratio {atr_ratio:.2f}x > 1.6 + out of squeeze")

    # ── REVERSAL: CHoCH + RSI extreme + weakening trend
    elif (bull_choch or bear_choch) and (rsi_val < 35 or rsi_val > 65) and adx_slope < 0:
        regime     = "REVERSAL"
        confidence = 0.65
        reasons.append(f"CHoCH detected + RSI {rsi_val:.1f} + ADX falling")

    # ── TREND_UP: ADX > 25 + +DI > -DI + EMA slope up
    elif adx_val > 25 and pdi_val > mdi_val and ema_slope > 0:
        regime     = "TREND_UP"
        # semakin tinggi ADX, semakin confident
        confidence = min(0.5 + (adx_val - 25) / 50, 0.95)
        reasons.append(f"ADX {adx_val:.1f} + +DI {pdi_val:.1f} > -DI {mdi_val:.1f}")

    # ── TREND_DOWN: ADX > 25 + -DI > +DI + EMA slope down
    elif adx_val > 25 and mdi_val > pdi_val and ema_slope < 0:
        regime     = "TREND_DOWN"
        confidence = min(0.5 + (adx_val - 25) / 50, 0.95)
        reasons.append(f"ADX {adx_val:.1f} + -DI {mdi_val:.1f} > +DI {pdi_val:.1f}")

    # ── RANGING: ADX < 20 atau BB squeeze
    elif adx_val < 20 or bb_squeeze:
        regime     = "RANGING"
        confidence = 0.5 + (20 - adx_val) / 40 if adx_val < 20 else 0.6
        reasons.append(f"ADX {adx_val:.1f} < 20" if adx_val < 20 else "BB squeeze")

    # ── TREND tapi lemah (ADX 20-25): lihat EMA slope
    else:
        if ema_slope > 0:
            regime     = "TREND_UP"
            confidence = 0.55
            reasons.append(f"Weak uptrend: ADX {adx_val:.1f}, EMA slope +")
        else:
            regime     = "TREND_DOWN"
            confidence = 0.55
            reasons.append(f"Weak downtrend: ADX {adx_val:.1f}, EMA slope -")

    return {
        "regime":      regime,
        "confidence":  round(confidence, 2),
        "adx":         round(adx_val, 2),
        "plus_di":     round(pdi_val, 2),
        "minus_di":    round(mdi_val, 2),
        "adx_slope":   round(adx_slope, 4),
        "bb_width":    round(bbw_now,  2),
        "bb_squeeze":  bb_squeeze,
        "volatility":  volatility,
        "atr_ratio":   round(atr_ratio, 2),
        "ema_slope":   round(ema_slope, 6),
        "price_pos":   round(price_pos, 2),
        "bull_choch":  bull_choch,
        "bear_choch":  bear_choch,
        "rsi":         round(rsi_val, 1),
        "emoji":       REGIME_LABELS[regime]["emoji"],
        "strategy":    REGIME_LABELS[regime]["strategy"],
        "reasons":     reasons,
    }

# ==================================================
# SCORE MODIFIER  (dipanggil dari advanced_score)
# ==================================================
def regime_score_modifier(regime: dict, action: str) -> int:
    """
    Return score modifier berdasarkan regime × action.
    Clamp ke [-15, +15].
    """
    key = (regime["regime"], action)
    mod = REGIME_SCORE_MOD.get(key, 0)

    # skala dengan confidence (0.5 = setengah, 1.0 = full)
    scaled = int(round(mod * regime["confidence"]))
    return max(-15, min(15, scaled))

# ==================================================
# SKIP FILTER  (dipanggil sebelum kirim signal)
# ==================================================
def should_skip(regime: dict, action: str, score: int) -> Tuple[bool, str]:
    """
    Return (should_skip, reason).
    True = jangan kirim signal ini.
    """
    r   = regime["regime"]
    rule = REGIME_SKIP_RULES.get(r)
    if rule and rule(action, score):
        return True, f"Regime {r}: score {score} terlalu rendah untuk {action}"
    return False, ""

# ==================================================
# EMBED FIELD  (helper untuk Discord embed)
# ==================================================
def regime_embed_value(regime: dict) -> str:
    """Format nilai untuk Discord embed field."""
    bb_sq = "🔵 SQUEEZE" if regime["bb_squeeze"] else "—"
    choch = ""
    if regime["bull_choch"]: choch = "⚡ Bull CHoCH"
    elif regime["bear_choch"]: choch = "⚡ Bear CHoCH"

    return (
        f"{regime['emoji']} **{regime['regime']}**  "
        f"conf: `{regime['confidence']*100:.0f}%`\n"
        f"ADX: `{regime['adx']:.1f}` "
        f"+DI: `{regime['plus_di']:.1f}` "
        f"-DI: `{regime['minus_di']:.1f}`\n"
        f"Volatility: `{regime['volatility']}`  "
        f"BBW: `{regime['bb_width']:.2f}%`  {bb_sq}\n"
        f"Strategy: `{regime['strategy']}`  {choch}"
    )

