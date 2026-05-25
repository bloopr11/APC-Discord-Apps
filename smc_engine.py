# ==================================================
# smc_engine.py  —  Smart Money Concepts Engine
#
# Implementasi SMC yang proper:
#   1. Market Structure  (BOS / CHoCH internal & external)
#   2. Displacement      (impulsive move validation)
#   3. Liquidity Sweep   (stop hunt detection)
#   4. Order Block       (institutional OB dengan mitigation check)
#   5. Inducement        (liquidity trap sebelum reversal)
#
# Cara pakai di app.py:
#   from smc_engine import smc_full, smc_score_modifier, smc_embed_value
#
#   # di generate_signal(), ganti:
#   #   smc_data = smc(df)
#   # dengan:
#   #   smc_data = smc_full(df)
#
#   # di advanced_score(), ganti cats["SMC"] dengan:
#   #   cats["SMC"] = smc_score_modifier(smc_data)
# ==================================================

import numpy as np
import pandas as pd
from typing import Optional


# ==================================================
# HELPERS
# ==================================================

def _swing_highs(high: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    n      = len(high)
    result = pd.Series(False, index=high.index)
    vals   = high.values
    for i in range(left, n - right):
        if all(vals[i] > vals[i - j] for j in range(1, left + 1)) and \
           all(vals[i] > vals[i + j] for j in range(1, right + 1)):
            result.iloc[i] = True
    return result


def _swing_lows(low: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    n      = len(low)
    result = pd.Series(False, index=low.index)
    vals   = low.values
    for i in range(left, n - right):
        if all(vals[i] < vals[i - j] for j in range(1, left + 1)) and \
           all(vals[i] < vals[i + j] for j in range(1, right + 1)):
            result.iloc[i] = True
    return result


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ==================================================
# 1. DISPLACEMENT DETECTION
#
# Displacement = candle impulsif yang menembus level
# dengan body besar. Ini yang membedakan real BOS
# dari fake breakout.
#
# Kriteria:
#   - Body candle > factor x ATR
#   - Body ratio > 50% dari range candle (bukan doji)
#   - Volume di atas rata-rata (jika tersedia)
# ==================================================

def detect_displacement(
    df: pd.DataFrame,
    idx: int,
    direction: str,
    atr_val: float,
    factor: float = 1.2,
) -> dict:
    o = float(df["Open"].iloc[idx])
    h = float(df["High"].iloc[idx])
    l = float(df["Low"].iloc[idx])
    c = float(df["Close"].iloc[idx])

    body         = abs(c - o)
    candle_range = h - l if h != l else 1e-10
    body_ratio   = body / candle_range

    is_bullish = c > o
    is_bearish = c < o
    solid_body = body > factor * atr_val and body_ratio > 0.5

    if direction == "UP":
        valid = solid_body and is_bullish
    else:
        valid = solid_body and is_bearish

    vol_confirm = True
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        avg_vol     = float(df["Volume"].iloc[max(0, idx-20):idx].mean())
        cur_vol     = float(df["Volume"].iloc[idx])
        vol_confirm = cur_vol > avg_vol * 0.9

    return {
        "valid":      valid and vol_confirm,
        "body":       body,
        "body_ratio": round(body_ratio, 3),
        "atr_val":    atr_val,
        "vol_ok":     vol_confirm,
    }


# ==================================================
# 2. LIQUIDITY SWEEP DETECTION
#
# Sweep terjadi saat harga mengambil liquidity di
# atas swing high / bawah swing low lalu berbalik.
#
# Kriteria:
#   - Wick menembus level swing sebelumnya
#   - Candle CLOSE kembali di sisi berlawanan
#   - Lower wick > 50% dari body (rejection)
# ==================================================

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 30) -> dict:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    open_ = df["Open"]
    n     = len(df)

    atr_now = float(_atr(df).iloc[-1])
    start   = max(5, n - lookback)

    bull_sweep = None
    bear_sweep = None

    for i in range(start, n):
        c_now = float(close.iloc[i])
        o_now = float(open_.iloc[i])
        h_now = float(high.iloc[i])
        l_now = float(low.iloc[i])

        # Bullish sweep: wick ke bawah swing low, close kembali di atas
        prev_lows     = low.iloc[max(0, i-20):i]
        swing_low_ref = float(prev_lows.min()) if len(prev_lows) > 0 else l_now

        if l_now < swing_low_ref and c_now > swing_low_ref:
            lower_wick = swing_low_ref - l_now
            body       = abs(c_now - o_now)
            if lower_wick > body * 0.5:
                bull_sweep = {
                    "idx":       i,
                    "level":     swing_low_ref,
                    "wick_size": round(lower_wick, 6),
                    "close":     c_now,
                    "bars_ago":  n - 1 - i,
                    "strength":  round(lower_wick / atr_now, 2) if atr_now > 0 else 0,
                }

        # Bearish sweep: wick ke atas swing high, close kembali di bawah
        prev_highs     = high.iloc[max(0, i-20):i]
        swing_high_ref = float(prev_highs.max()) if len(prev_highs) > 0 else h_now

        if h_now > swing_high_ref and c_now < swing_high_ref:
            upper_wick = h_now - swing_high_ref
            body       = abs(c_now - o_now)
            if upper_wick > body * 0.5:
                bear_sweep = {
                    "idx":       i,
                    "level":     swing_high_ref,
                    "wick_size": round(upper_wick, 6),
                    "close":     c_now,
                    "bars_ago":  n - 1 - i,
                    "strength":  round(upper_wick / atr_now, 2) if atr_now > 0 else 0,
                }

    fresh_bull = bull_sweep is not None and bull_sweep["bars_ago"] <= 5
    fresh_bear = bear_sweep is not None and bear_sweep["bars_ago"] <= 5

    return {
        "bull_sweep": bull_sweep,
        "bear_sweep": bear_sweep,
        "fresh_bull": fresh_bull,
        "fresh_bear": fresh_bear,
        "has_sweep":  bull_sweep is not None or bear_sweep is not None,
    }


# ==================================================
# 3. MARKET STRUCTURE  (BOS & CHoCH)
#
# EXTERNAL structure: swing high/low major (left=3, right=3)
# INTERNAL structure: minor swing (left=2, right=2)
#
# BOS  (Break of Structure):
#   - Close menembus swing high/low terakhir
#   - Disertai displacement candle
#
# CHoCH (Change of Character):
#   - Dalam uptrend: close bawah swing low internal
#   - Dalam downtrend: close atas swing high internal
#   - Lebih valid jika didahului liquidity sweep
#
# HH/HL/LH/LL dari 3 swing terakhir → structure label
# ==================================================

def detect_market_structure(
    df: pd.DataFrame,
    swing_left:  int = 3,
    swing_right: int = 3,
    lookback:    int = 50,
) -> dict:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    n     = len(df)

    atr_now = float(_atr(df).iloc[-1])

    sh_mask   = _swing_highs(high, swing_left, swing_right)
    sl_mask   = _swing_lows(low,   swing_left, swing_right)
    start     = max(0, n - lookback)

    sh_recent = [i for i in range(start, n) if sh_mask.iloc[i] and i < n - swing_right]
    sl_recent = [i for i in range(start, n) if sl_mask.iloc[i] and i < n - swing_right]

    price_now = float(close.iloc[-1])

    # ── BOS Bullish ───────────────────────────────
    bull_bos = False; bull_bos_level = None; bull_bos_disp = False
    if sh_recent:
        last_sh = sh_recent[-1]
        last_sh_level = float(high.iloc[last_sh])
        if price_now > last_sh_level:
            bull_bos = True
            bull_bos_level = last_sh_level
            for i in range(last_sh + 1, n):
                if float(close.iloc[i]) > last_sh_level:
                    if detect_displacement(df, i, "UP", atr_now)["valid"]:
                        bull_bos_disp = True
                    break

    # ── BOS Bearish ───────────────────────────────
    bear_bos = False; bear_bos_level = None; bear_bos_disp = False
    if sl_recent:
        last_sl = sl_recent[-1]
        last_sl_level = float(low.iloc[last_sl])
        if price_now < last_sl_level:
            bear_bos = True
            bear_bos_level = last_sl_level
            for i in range(last_sl + 1, n):
                if float(close.iloc[i]) < last_sl_level:
                    if detect_displacement(df, i, "DOWN", atr_now)["valid"]:
                        bear_bos_disp = True
                    break

    # ── CHoCH ─────────────────────────────────────
    ema50      = close.ewm(span=50, adjust=False).mean()
    in_uptrend = price_now > float(ema50.iloc[-1])

    choch_bull = False; choch_bear = False; choch_level = None
    if in_uptrend and sl_recent:
        int_sl_level = float(low.iloc[sl_recent[-1]])
        if price_now < int_sl_level:
            choch_bear = True; choch_level = int_sl_level
    elif not in_uptrend and sh_recent:
        int_sh_level = float(high.iloc[sh_recent[-1]])
        if price_now > int_sh_level:
            choch_bull = True; choch_level = int_sh_level

    # ── HH / HL / LH / LL ────────────────────────
    structure_label = "UNCLEAR"
    if len(sh_recent) >= 2 and len(sl_recent) >= 2:
        hh = float(high.iloc[sh_recent[-1]]) > float(high.iloc[sh_recent[-2]])
        hl = float(low.iloc[sl_recent[-1]])  > float(low.iloc[sl_recent[-2]])
        lh = float(high.iloc[sh_recent[-1]]) < float(high.iloc[sh_recent[-2]])
        ll = float(low.iloc[sl_recent[-1]])  < float(low.iloc[sl_recent[-2]])

        if hh and hl:    structure_label = "BULLISH (HH+HL)"
        elif lh and ll:  structure_label = "BEARISH (LH+LL)"
        else:            structure_label = "MIXED"

    return {
        "bull_bos":       bull_bos,
        "bull_bos_level": bull_bos_level,
        "bull_bos_disp":  bull_bos_disp,
        "bear_bos":       bear_bos,
        "bear_bos_level": bear_bos_level,
        "bear_bos_disp":  bear_bos_disp,
        "choch_bull":     choch_bull,
        "choch_bear":     choch_bear,
        "choch_level":    choch_level,
        "structure":      structure_label,
        "swing_highs":    [(i, float(high.iloc[i])) for i in sh_recent[-5:]],
        "swing_lows":     [(i, float(low.iloc[i]))  for i in sl_recent[-5:]],
    }


# ==================================================
# 4. ORDER BLOCK DETECTION  (Institutional)
#
# Institutional OB = last opposing candle sebelum
# impulse move yang kuat.
#
# Bull OB: candle bearish terakhir sebelum impulse naik
# Bear OB: candle bullish terakhir sebelum impulse turun
#
# Valid jika:
#   - Belum dimitigasi (harga belum masuk > mitig_pct)
#   - Candle setelah OB adalah displacement
#   - Semakin strength tinggi, semakin kuat OB
# ==================================================

def detect_order_blocks(
    df: pd.DataFrame,
    lookback:  int   = 50,
    mitig_pct: float = 0.5,
) -> dict:
    high  = df["High"].values
    low   = df["Low"].values
    close = df["Close"].values
    open_ = df["Open"].values
    n     = len(df)

    atr_now = float(_atr(df).iloc[-1])
    start   = max(3, n - lookback)

    bull_obs = []
    bear_obs = []

    for i in range(start, n - 2):
        # ── Bullish OB ────────────────────────────
        if close[i] < open_[i]:  # candle bearish
            next_body = abs(close[i+1] - open_[i+1])
            if close[i+1] > open_[i+1] and next_body > atr_now * 0.8:
                ob_top    = max(open_[i], close[i])
                ob_bottom = min(open_[i], close[i])
                mitigated = False
                for p in close[i+2:]:
                    pen = (ob_top - p) / (ob_top - ob_bottom + 1e-10)
                    if p <= ob_top and pen >= mitig_pct:
                        mitigated = True; break
                price_now = close[-1]
                if not mitigated:
                    bull_obs.append({
                        "idx":      i,
                        "top":      round(float(ob_top),    6),
                        "bottom":   round(float(ob_bottom), 6),
                        "mid":      round(float((ob_top + ob_bottom) / 2), 6),
                        "in_zone":  float(ob_bottom) <= price_now <= float(ob_top),
                        "bars_ago": n - 1 - i,
                        "strength": round(next_body / atr_now, 2),
                    })

        # ── Bearish OB ────────────────────────────
        if close[i] > open_[i]:  # candle bullish
            next_body = abs(close[i+1] - open_[i+1])
            if close[i+1] < open_[i+1] and next_body > atr_now * 0.8:
                ob_top    = max(open_[i], close[i])
                ob_bottom = min(open_[i], close[i])
                mitigated = False
                for p in close[i+2:]:
                    pen = (p - ob_bottom) / (ob_top - ob_bottom + 1e-10)
                    if p >= ob_bottom and pen >= mitig_pct:
                        mitigated = True; break
                price_now = close[-1]
                if not mitigated:
                    bear_obs.append({
                        "idx":      i,
                        "top":      round(float(ob_top),    6),
                        "bottom":   round(float(ob_bottom), 6),
                        "mid":      round(float((ob_top + ob_bottom) / 2), 6),
                        "in_zone":  float(ob_bottom) <= price_now <= float(ob_top),
                        "bars_ago": n - 1 - i,
                        "strength": round(next_body / atr_now, 2),
                    })

    price_now = float(close[-1])
    below     = [o for o in bull_obs if o["top"] <= price_now]
    above     = [o for o in bear_obs if o["bottom"] >= price_now]

    nearest_bull_ob = min(below, key=lambda x: price_now - x["top"]) if below else None
    nearest_bear_ob = min(above, key=lambda x: x["bottom"] - price_now) if above else None

    return {
        "bull_obs":        bull_obs[-5:],
        "bear_obs":        bear_obs[-5:],
        "nearest_bull_ob": nearest_bull_ob,
        "nearest_bear_ob": nearest_bear_ob,
        "in_bull_ob":      any(o["in_zone"] for o in bull_obs),
        "in_bear_ob":      any(o["in_zone"] for o in bear_obs),
        "count_bull":      len(bull_obs),
        "count_bear":      len(bear_obs),
    }


# ==================================================
# 5. INDUCEMENT DETECTION
#
# Liquidity trap yang sengaja dibuat sebelum
# smart money mengambil posisi.
# Minor swing yang belum disweep = target likuiditas.
# ==================================================

def detect_inducement(df: pd.DataFrame, lookback: int = 20) -> dict:
    high  = df["High"]
    low   = df["Low"]
    close = df["Close"]
    n     = len(df)

    sh_mask     = _swing_highs(high, left=2, right=2)
    sl_mask     = _swing_lows(low,   left=2, right=2)
    start       = max(0, n - lookback)

    sh_internal = [i for i in range(start, n) if sh_mask.iloc[i]]
    sl_internal = [i for i in range(start, n) if sl_mask.iloc[i]]
    price_now   = float(close.iloc[-1])

    ind_bear = [
        {"idx": i, "level": float(high.iloc[i]), "bars_ago": n - 1 - i}
        for i in sh_internal
        if float(high.iloc[i]) > price_now and (n - 1 - i) <= 10
    ]
    ind_bull = [
        {"idx": i, "level": float(low.iloc[i]), "bars_ago": n - 1 - i}
        for i in sl_internal
        if float(low.iloc[i]) < price_now and (n - 1 - i) <= 10
    ]

    return {
        "bear_targets": ind_bear,
        "bull_targets": ind_bull,
        "has_bear_ind": len(ind_bear) > 0,
        "has_bull_ind": len(ind_bull) > 0,
        "nearest_bear": ind_bear[0]  if ind_bear else None,
        "nearest_bull": ind_bull[-1] if ind_bull else None,
    }


# ==================================================
# 6. FULL SMC PIPELINE
# ==================================================

def smc_full(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Entry point utama. Panggil ini dari generate_signal()
    sebagai pengganti smc(df) yang lama.
    """
    atr_now   = float(_atr(df).iloc[-1])
    structure = detect_market_structure(df, lookback=lookback)
    sweep     = detect_liquidity_sweep(df,  lookback=lookback)
    ob        = detect_order_blocks(df,     lookback=lookback)
    indu      = detect_inducement(df,       lookback=20)

    # Konfirmasi silang
    confirmed_bull_bos = (
        structure["bull_bos"] and structure["bull_bos_disp"] and
        (sweep["fresh_bull"] or sweep["bull_sweep"] is not None)
    )
    confirmed_bear_bos = (
        structure["bear_bos"] and structure["bear_bos_disp"] and
        (sweep["fresh_bear"] or sweep["bear_sweep"] is not None)
    )
    confirmed_choch_bull = structure["choch_bull"] and sweep["fresh_bull"]
    confirmed_choch_bear = structure["choch_bear"] and sweep["fresh_bear"]

    ideal_bull = confirmed_bull_bos and ob["in_bull_ob"] and sweep["fresh_bull"]
    ideal_bear = confirmed_bear_bos and ob["in_bear_ob"] and sweep["fresh_bear"]

    # Backward compat dengan advanced_score() yang pakai keys lama
    return {
        # ── Legacy keys (kompatibel dengan advanced_score) ──
        "BOS":         confirmed_bull_bos or confirmed_bear_bos,
        "CHoCH":       confirmed_choch_bull or confirmed_choch_bear,
        "LIQ":         sweep["has_sweep"],
        "ORDER_BLOCK": ob["in_bull_ob"] or ob["in_bear_ob"],

        # ── Structure ───────────────────────────────────────
        "structure":              structure,
        "bull_bos":               structure["bull_bos"],
        "bull_bos_confirmed":     confirmed_bull_bos,
        "bear_bos":               structure["bear_bos"],
        "bear_bos_confirmed":     confirmed_bear_bos,
        "choch_bull":             structure["choch_bull"],
        "choch_bear":             structure["choch_bear"],
        "choch_bull_confirmed":   confirmed_choch_bull,
        "choch_bear_confirmed":   confirmed_choch_bear,
        "structure_label":        structure["structure"],

        # ── Sweep ───────────────────────────────────────────
        "sweep":                  sweep,
        "fresh_bull_sweep":       sweep["fresh_bull"],
        "fresh_bear_sweep":       sweep["fresh_bear"],

        # ── Order Block ─────────────────────────────────────
        "ob":                     ob,
        "in_bull_ob":             ob["in_bull_ob"],
        "in_bear_ob":             ob["in_bear_ob"],
        "nearest_bull_ob":        ob["nearest_bull_ob"],
        "nearest_bear_ob":        ob["nearest_bear_ob"],

        # ── Inducement ──────────────────────────────────────
        "inducement":             indu,
        "has_bull_ind":           indu["has_bull_ind"],
        "has_bear_ind":           indu["has_bear_ind"],

        # ── Setup quality ───────────────────────────────────
        "ideal_bull_setup":       ideal_bull,
        "ideal_bear_setup":       ideal_bear,
        "atr":                    atr_now,
    }


# ==================================================
# 7. SCORE MODIFIER  (-20 s/d +20)
# Menggantikan cats["SMC"] di advanced_score()
# ==================================================

def smc_score_modifier(smc_data: dict) -> int:
    s = 0

    if smc_data.get("bull_bos_confirmed"):   s += 10
    elif smc_data.get("bull_bos"):           s += 4
    if smc_data.get("bear_bos_confirmed"):   s -= 10
    elif smc_data.get("bear_bos"):           s -= 4

    if smc_data.get("choch_bull_confirmed"): s += 7
    elif smc_data.get("choch_bull"):         s += 3
    if smc_data.get("choch_bear_confirmed"): s -= 7
    elif smc_data.get("choch_bear"):         s -= 3

    if smc_data.get("fresh_bull_sweep"):     s += 5
    if smc_data.get("fresh_bear_sweep"):     s -= 5

    if smc_data.get("in_bull_ob"):           s += 6
    if smc_data.get("in_bear_ob"):           s -= 6

    if smc_data.get("ideal_bull_setup"):     s += 5
    if smc_data.get("ideal_bear_setup"):     s -= 5

    if smc_data.get("has_bear_ind"):         s -= 3
    if smc_data.get("has_bull_ind"):         s += 3

    return max(-20, min(20, s))


# ==================================================
# 8. EMBED VALUE
# ==================================================

def smc_embed_value(smc_data: dict) -> str:
    structure = smc_data.get("structure_label", "UNCLEAR")

    if smc_data.get("bull_bos_confirmed"):
        bos_str = "🟢 Bull BOS ✅ displaced"
    elif smc_data.get("bull_bos"):
        bos_str = "🟢 Bull BOS ⚠️ no displacement"
    elif smc_data.get("bear_bos_confirmed"):
        bos_str = "🔴 Bear BOS ✅ displaced"
    elif smc_data.get("bear_bos"):
        bos_str = "🔴 Bear BOS ⚠️ no displacement"
    else:
        bos_str = "➖ No BOS"

    choch_str = None
    if smc_data.get("choch_bull_confirmed"):  choch_str = "🔄 CHoCH Bull ✅"
    elif smc_data.get("choch_bear_confirmed"): choch_str = "🔄 CHoCH Bear ✅"
    elif smc_data.get("choch_bull") or smc_data.get("choch_bear"):
        choch_str = "🔄 CHoCH (unconfirmed)"

    sweep = smc_data.get("sweep", {})
    sweep_str = None
    if sweep.get("fresh_bull"):
        s = sweep.get("bull_sweep") or {}
        sweep_str = f"💧 Bull Sweep `{s.get('strength', 0):.1f}x` ATR"
    elif sweep.get("fresh_bear"):
        s = sweep.get("bear_sweep") or {}
        sweep_str = f"💧 Bear Sweep `{s.get('strength', 0):.1f}x` ATR"

    ob_str = None
    if smc_data.get("in_bull_ob"):
        nb = smc_data.get("nearest_bull_ob") or {}
        ob_str = f"📦 IN Bull OB `{nb.get('bottom',0):.4f}`-`{nb.get('top',0):.4f}` str:`{nb.get('strength',0):.1f}x`"
    elif smc_data.get("in_bear_ob"):
        nb = smc_data.get("nearest_bear_ob") or {}
        ob_str = f"📦 IN Bear OB `{nb.get('bottom',0):.4f}`-`{nb.get('top',0):.4f}` str:`{nb.get('strength',0):.1f}x`"

    setup_str = None
    if smc_data.get("ideal_bull_setup"):   setup_str = "⭐ IDEAL BULL SETUP"
    elif smc_data.get("ideal_bear_setup"): setup_str = "⭐ IDEAL BEAR SETUP"

    parts = [
        f"Structure: `{structure}`",
        bos_str,
        choch_str,
        sweep_str,
        ob_str,
        setup_str,
    ]
    return "\n".join(p for p in parts if p)