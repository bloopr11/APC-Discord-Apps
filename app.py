import discord
from discord import app_commands
from discord.ui import View, Button, Select
import asyncio
import os
import functools
import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib
matplotlib.use("Agg")

# ==================================================
# ENV
# ==================================================
TOKEN      = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

# ==================================================
# PAIRS
# ==================================================
PAIRS = {
    "BTCUSD":  "BTC-USD",
    "ETHUSD":  "ETH-USD",
    "XAUUSD":  "GC=F",
    "XRPUSD":  "XRP-USD",
    "SOLUSD":  "SOL-USD",
    "ADAUSD":  "ADA-USD",
    "DOGEUSD": "DOGE-USD",
    "BNBUSD":  "BNB-USD",
    "AVAXUSD": "AVAX-USD",
    "LINKUSD": "LINK-USD",
}

TIMEFRAMES = {
    "5m":  {"period": "5d",  "interval": "5m"},
    "15m": {"period": "7d",  "interval": "15m"},
    "1h":  {"period": "30d", "interval": "1h"},
    "4h":  {"period": "60d", "interval": "4h"},
}

DEFAULT_TF  = "5m"
AUTO_SIGNAL = False
AUTO_MIN_SCORE = 75          # kirim auto-signal jika score >= nilai ini

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
# ASYNC HELPER — jalankan fungsi blocking di thread pool
# agar event loop Discord tidak freeze saat download/chart
# ==================================================
async def run_blocking(func, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, functools.partial(func, *args, **kwargs)
    )

# ==================================================
# MARKET DATA
# ==================================================
def get_data(symbol: str, timeframe: str = DEFAULT_TF) -> pd.DataFrame:
    cfg = TIMEFRAMES.get(timeframe, TIMEFRAMES[DEFAULT_TF])
    df  = yf.download(
        symbol,
        period   = cfg["period"],
        interval = cfg["interval"],
        auto_adjust = True,
        progress    = False,
    )
    df = df.dropna()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df

# ==================================================
# INDICATORS
# ==================================================
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
    sma    = series.rolling(period).mean()
    std    = series.rolling(period).std()
    upper  = sma + std_dev * std
    lower  = sma - std_dev * std
    return upper, sma, lower

def volume_ratio(df: pd.DataFrame, period: int = 20) -> float:
    """Current volume vs rolling average — >1 means above-average activity."""
    if "Volume" not in df.columns or df["Volume"].sum() == 0:
        return 1.0
    avg_vol = df["Volume"].rolling(period).mean().iloc[-1]
    cur_vol = df["Volume"].iloc[-1]
    return float(cur_vol / avg_vol) if avg_vol else 1.0

# ==================================================
# RSI DIVERGENCE
# ==================================================
def rsi_divergence(df: pd.DataFrame, rsi_series: pd.Series, lookback: int = 5) -> str:
    """
    Bullish divergence  : price lower low, RSI higher low  → potential reversal up
    Bearish divergence  : price higher high, RSI lower high → potential reversal down
    """
    close = df["Close"]
    price_diff = close.iloc[-1] - close.iloc[-lookback]
    rsi_diff   = rsi_series.iloc[-1] - rsi_series.iloc[-lookback]

    if price_diff < 0 and rsi_diff > 0:
        return "BULLISH_DIV"
    if price_diff > 0 and rsi_diff < 0:
        return "BEARISH_DIV"
    return "NONE"

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

    bullish = price > e8 > e21 > e50    # stacked bull
    bearish = price < e8 < e21 < e50    # stacked bear
    near_200 = abs(price - e200) / e200 < 0.005   # within 0.5% of 200 EMA

    return {
        "BULLISH_STACK": bullish,
        "BEARISH_STACK": bearish,
        "NEAR_200EMA":   near_200,
        "E8": e8, "E21": e21, "E50": e50, "E200": e200,
    }

# ==================================================
# SMC ENGINE (improved)
# ==================================================
def smc(df: pd.DataFrame) -> dict:
    close, high, low = df["Close"], df["High"], df["Low"]

    swing_high  = high.rolling(10).max()
    swing_low   = low.rolling(10).min()

    bos    = close.iloc[-1] > swing_high.iloc[-2]
    choch  = close.iloc[-1] < swing_low.iloc[-2]
    liq    = high.iloc[-1]  > high.rolling(20).max().iloc[-2]

    # Order block: last big bearish/bullish candle before a BOS
    body       = (close - df["Open"]).abs()
    big_candle = body.iloc[-3] > body.rolling(10).mean().iloc[-3]

    return {
        "BOS":        bool(bos),
        "CHoCH":      bool(choch),
        "LIQ":        bool(liq),
        "ORDER_BLOCK": bool(big_candle),
    }

# ==================================================
# LIQUIDITY ENGINE
# ==================================================
def liquidity(df: pd.DataFrame) -> dict:
    high_zone     = df["High"].rolling(20).max().iloc[-1]
    low_zone      = df["Low"].rolling(20).min().iloc[-1]
    current_price = df["Close"].iloc[-1]
    pressure      = (
        "BUY" if abs(current_price - low_zone) < abs(current_price - high_zone)
        else "SELL"
    )
    return {"HIGH_ZONE": high_zone, "LOW_ZONE": low_zone, "PRESSURE": pressure}

# ==================================================
# FLOW ENGINE
# ==================================================
def flow(df: pd.DataFrame) -> dict:
    momentum = df["Close"].diff().mean()
    return {
        "FLOW":     "BUY" if momentum > 0 else "SELL",
        "STRENGTH": abs(momentum) * 100,
    }

# ==================================================
# LSTM PROXY (improved: multi-EMA weighted bias)
# ==================================================
def lstm_prediction(df: pd.DataFrame) -> str:
    close   = df["Close"]
    price   = close.iloc[-1]
    ema20   = ema(close, 20).iloc[-1]
    ema50   = ema(close, 50).iloc[-1]
    ema200  = ema(close, 200).iloc[-1]
    score   = sum([price > ema20, price > ema50, price > ema200])
    return "BULLISH" if score >= 2 else "BEARISH"

# ==================================================
# IMPROVED AI SCORE  (max 100)
# ==================================================
def ai_score(lstm, smc_data, liq, fl, rsi_val, macd_hist, ema_conf,
             rsi_div, vol_ratio, stoch_k_val, bb_pos) -> int:
    score = 50

    # ── LSTM (multi-EMA) ──────────────────────────
    score += 15 if lstm == "BULLISH" else -15

    # ── SMC ──────────────────────────────────────
    if smc_data["BOS"]:         score += 8
    if smc_data["LIQ"]:         score += 6
    if smc_data["ORDER_BLOCK"]: score += 5
    if smc_data["CHoCH"]:       score -= 10

    # ── Liquidity pressure ────────────────────────
    score += 8 if liq["PRESSURE"] == "BUY" else -8

    # ── Flow ─────────────────────────────────────
    score += 5 if fl["FLOW"] == "BUY" else -5

    # ── RSI ──────────────────────────────────────
    if rsi_val < 30:   score += 10   # oversold → bullish edge
    elif rsi_val > 70: score -= 10   # overbought → bearish edge
    elif 45 <= rsi_val <= 55:
        score += 0                   # neutral zone
    elif rsi_val > 55: score += 4
    else:              score -= 4

    # ── RSI Divergence ────────────────────────────
    if rsi_div == "BULLISH_DIV":  score += 8
    elif rsi_div == "BEARISH_DIV": score -= 8

    # ── MACD Histogram ────────────────────────────
    if macd_hist > 0:  score += 6
    else:              score -= 6

    # ── EMA Confluence ───────────────────────────
    if ema_conf["BULLISH_STACK"]:  score += 10
    if ema_conf["BEARISH_STACK"]:  score -= 10
    if ema_conf["NEAR_200EMA"]:    score += 3   # 200 EMA magnet

    # ── Volume ───────────────────────────────────
    if vol_ratio >= 1.5:   score += 5   # high volume confirms move
    elif vol_ratio < 0.7:  score -= 3   # weak volume → less reliable

    # ── Stochastic ───────────────────────────────
    if stoch_k_val < 20:   score += 5
    elif stoch_k_val > 80: score -= 5

    # ── Bollinger Band position ───────────────────
    # bb_pos: 0 = at lower, 1 = at upper
    if bb_pos < 0.15:      score += 4   # near lower band
    elif bb_pos > 0.85:    score -= 4   # near upper band

    return max(0, min(100, score))

# ==================================================
# CONFIDENCE LABEL
# ==================================================
def confidence_label(score: int) -> str:
    if score >= 80: return "🔥 VERY HIGH"
    if score >= 65: return "✅ HIGH"
    if score >= 50: return "⚠️ MODERATE"
    return "❌ LOW"

# ==================================================
# LIQUIDITY POOL ZONES  (swing highs / swing lows)
# ==================================================
def liquidity_pools(df: pd.DataFrame, swing: int = 20):
    """
    Kembalikan zona likuiditas utama:
    - buy_side_liq  : kumpulan stop-loss BUY yang tersimpan di bawah swing low
                      (target stop-hunt oleh market maker saat harga turun)
    - sell_side_liq : kumpulan stop-loss SELL yang tersimpan di atas swing high
                      (target stop-hunt saat harga naik)
    Masing-masing berupa (level_harga, buffer_bawah, buffer_atas)
    """
    high = df["High"]
    low  = df["Low"]
    atr_now = float(df["High"].rolling(14).mean().iloc[-1] -
                    df["Low"].rolling(14).mean().iloc[-1])

    # Swing low = titik terendah dalam window → di sinilah SL retail BUY menumpuk
    swing_low  = float(low.rolling(swing).min().iloc[-1])
    # Swing high = titik tertinggi → di sinilah SL retail SELL menumpuk
    swing_high = float(high.rolling(swing).max().iloc[-1])

    # Buffer kecil di luar pool agar SL kita melewati zona sweep
    buf = atr_now * 0.3

    return {
        "buy_side_liq":  swing_low,   # pool SL di bawah → rawan di-sweep ke bawah
        "sell_side_liq": swing_high,  # pool SL di atas  → rawan di-sweep ke atas
        "buf":           buf,
        "swing":         swing,
    }

# ==================================================
# ANTI-LIQUIDITY-SWEEP TP / SL
# ==================================================
def tp_sl_anti_sweep(df: pd.DataFrame, action: str, atr_val: float):
    """
    SL ditempatkan BEYOND liquidity pool, bukan di dalam pool.
    Artinya SL kita sudah di luar zona di mana market maker biasanya
    melakukan stop-hunt, sehingga tidak mudah tersapu (sweep).

    BUY  → SL di bawah swing_low - buffer  (melewati pool SL retail)
    SELL → SL di atas  swing_high + buffer  (melewati pool SL retail)

    TP menggunakan 2.5× jarak SL dari entry agar RR minimal 1:2.5
    """
    price  = float(df["Close"].iloc[-1])
    pools  = liquidity_pools(df)

    if action == "BUY":
        # SL: di bawah liquidity pool bawah
        raw_sl    = pools["buy_side_liq"] - pools["buf"]
        # pastikan SL tidak terlalu jauh (max 3× ATR) — cegah SL absurd
        sl        = max(raw_sl, price - 3.0 * atr_val)
        sl_dist   = abs(price - sl)
        tp        = price + 2.5 * sl_dist   # RR 1:2.5
        sweep_ref = pools["buy_side_liq"]
        sweep_dir = "below"
    else:
        # SL: di atas liquidity pool atas
        raw_sl    = pools["sell_side_liq"] + pools["buf"]
        sl        = min(raw_sl, price + 3.0 * atr_val)
        sl_dist   = abs(price - sl)
        tp        = price - 2.5 * sl_dist
        sweep_ref = pools["sell_side_liq"]
        sweep_dir = "above"

    rr = abs(tp - price) / sl_dist if sl_dist > 0 else 0

    return {
        "tp":         tp,
        "sl":         sl,
        "rr":         rr,
        "sl_dist":    sl_dist,
        "sweep_ref":  sweep_ref,   # level pool likuiditas yang dijadikan acuan
        "sweep_dir":  sweep_dir,   # "above" / "below"
        "buf":        pools["buf"],
        "buy_pool":   pools["buy_side_liq"],
        "sell_pool":  pools["sell_side_liq"],
    }

# ==================================================
# GENERATE SIGNAL
# ==================================================
def generate_signal(pair: str, timeframe: str = DEFAULT_TF) -> dict:
    df    = get_data(PAIRS[pair], timeframe)
    close = df["Close"]

    # indicators
    rsi_series          = rsi(close)
    rsi_val             = float(rsi_series.iloc[-1])
    macd_line, macd_sig, macd_hist = macd(close)
    macd_hist_val       = float(macd_hist.iloc[-1])
    atr_val             = float(atr(df).iloc[-1])
    stoch_k, stoch_d    = stochastic(df)
    stoch_k_val         = float(stoch_k.iloc[-1])
    bb_upper, bb_mid, bb_lower = bollinger_bands(close)
    bb_range            = float(bb_upper.iloc[-1] - bb_lower.iloc[-1])
    bb_pos              = float(
        (close.iloc[-1] - bb_lower.iloc[-1]) / bb_range
        if bb_range else 0.5
    )
    vol_ratio_val       = volume_ratio(df)
    rsi_div             = rsi_divergence(df, rsi_series)
    ema_conf            = ema_confluence(df)

    # SMC / Liq / Flow / LSTM
    smc_data = smc(df)
    liq      = liquidity(df)
    fl       = flow(df)
    lstm     = lstm_prediction(df)

    score  = ai_score(
        lstm, smc_data, liq, fl, rsi_val, macd_hist_val,
        ema_conf, rsi_div, vol_ratio_val, stoch_k_val, bb_pos
    )
    action   = "BUY" if score >= 55 else "SELL"
    sl_data  = tp_sl_anti_sweep(df, action, atr_val)
    tp       = sl_data["tp"]
    sl       = sl_data["sl"]
    price    = float(close.iloc[-1])
    rr       = sl_data["rr"]

    return {
        "pair":       pair,
        "timeframe":  timeframe,
        "score":      score,
        "confidence": confidence_label(score),
        "action":     action,
        "entry":      price,
        "tp":         tp,
        "sl":         sl,
        "rr":         rr,
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
        "sweep":      sl_data,   # anti-sweep detail
        "df":         df,
    }

# ==================================================
# FORMAT SIGNAL (Discord Embed)
# ==================================================
def build_embed(data: dict) -> discord.Embed:
    action_color = discord.Color.green() if data["action"] == "BUY" else discord.Color.red()
    action_emoji = "🟢" if data["action"] == "BUY" else "🔴"
    sw           = data["sweep"]

    embed = discord.Embed(
        title       = f"{action_emoji} {data['pair']} — {data['action']} SIGNAL",
        description = f"**AI Score:** `{data['score']}%` — {data['confidence']}",
        color       = action_color,
    )
    embed.add_field(
        name  = "📍 Entry / 🎯 TP / 🛑 SL",
        value = (
            f"`{data['entry']:.4f}` / `{data['tp']:.4f}` / `{data['sl']:.4f}`\n"
            f"**R:R** → `1:{data['rr']:.2f}`"
        ),
        inline=False,
    )

    # ── Anti-Sweep SL detail ──────────────────────
    sweep_side = "Buy-Side Pool (Swing Low)" if data["action"] == "BUY" else "Sell-Side Pool (Swing High)"
    sweep_pos  = "di bawah" if data["action"] == "BUY" else "di atas"
    embed.add_field(
        name  = "🛡️ Anti Liquidity Sweep SL",
        value = (
            f"**Liquidity Pool:** `{sw['sweep_ref']:.4f}` ({sweep_side})\n"
            f"**Buffer beyond pool:** `{sw['buf']:.4f}`\n"
            f"**SL ditempatkan {sweep_pos} pool** → `{data['sl']:.4f}`\n"
            f"*SL berada di luar zona stop-hunt market maker,\n"
            f"sehingga tidak mudah tersapu (sweep) sebelum harga bergerak.*"
        ),
        inline=False,
    )
    embed.add_field(
        name  = "💧 Liquidity Zones",
        value = (
            f"Buy-Side Pool  (Swing Low):  `{sw['buy_pool']:.4f}`\n"
            f"Sell-Side Pool (Swing High): `{sw['sell_pool']:.4f}`"
        ),
        inline=False,
    )
    embed.add_field(
        name  = "📊 Indicators",
        value = (
            f"RSI: `{data['rsi']:.1f}` | MACD Hist: `{data['macd']:.4f}`\n"
            f"Stoch K: `{data['stoch_k']:.1f}` | ATR: `{data['atr']:.4f}`\n"
            f"Vol Ratio: `{data['vol_ratio']:.2f}x` | BB Pos: `{data['bb_pos']:.2f}`"
        ),
        inline=False,
    )
    div_text = {"BULLISH_DIV": "🔼 Bullish", "BEARISH_DIV": "🔽 Bearish", "NONE": "—"}
    embed.add_field(
        name  = "🧠 AI Analysis",
        value = (
            f"LSTM: `{data['lstm']}` | RSI Div: `{div_text[data['rsi_div']]}`\n"
            f"EMA Stack: `{'🐂 Bull' if data['ema']['BULLISH_STACK'] else '🐻 Bear' if data['ema']['BEARISH_STACK'] else 'Mixed'}`\n"
            f"BOS: `{data['smc']['BOS']}` | CHoCH: `{data['smc']['CHoCH']}` | OB: `{data['smc']['ORDER_BLOCK']}`\n"
            f"Liq Pressure: `{data['liq']['PRESSURE']}` | Flow: `{data['flow']['FLOW']}`"
        ),
        inline=False,
    )
    embed.set_footer(text=f"⏱ TF: {data['timeframe'].upper()} | Powered by AI Institutional Engine")
    return embed

# ==================================================
# CHART  — matplotlib candle chart dengan TP/SL zone
# ==================================================
def create_chart(
    df: pd.DataFrame,
    pair: str,
    entry: float,
    tp: float,
    sl: float,
    timeframe: str = DEFAULT_TF,
    sweep_data: dict = None,   # dari tp_sl_anti_sweep()
) -> str:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    df_c = df.copy()

    # ── pastikan DatetimeIndex ────────────────────
    if not isinstance(df_c.index, pd.DatetimeIndex):
        df_c.index = pd.to_datetime(df_c.index)

    # ── trim ke jumlah candle per TF ─────────────
    candle_limit = {"5m": 100, "15m": 100, "1h": 96, "4h": 60}
    n    = candle_limit.get(timeframe, 100)
    df_c = df_c.iloc[-n:]

    # ── EMA ──────────────────────────────────────
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

    # ── figure: 2 baris (chart + info panel) ─────
    fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
    gs  = fig.add_gridspec(
        2, 1, height_ratios=[5, 1],
        hspace=0.06, left=0.01, right=0.86, top=0.93, bottom=0.08,
    )
    ax  = fig.add_subplot(gs[0])   # candle chart
    ax2 = fig.add_subplot(gs[1])   # info bar bawah

    ax.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")
    ax2.axis("off")

    # ═══════════════════════════════════════════════
    # LAYER 0 — Liquidity Pool Zones (jika ada)
    # ═══════════════════════════════════════════════
    if sweep_data:
        buy_pool  = sweep_data["buy_pool"]
        sell_pool = sweep_data["sell_pool"]
        buf       = sweep_data["buf"]

        # Buy-side liquidity pool (kuning transparan, bawah chart)
        ax.axhspan(
            buy_pool - buf * 2, buy_pool,
            alpha=0.08, color="#ffd600", zorder=1,
        )
        ax.axhline(
            buy_pool, color="#ffd600", linewidth=0.9,
            linestyle=":", zorder=2, alpha=0.7,
        )

        # Sell-side liquidity pool (ungu transparan, atas chart)
        ax.axhspan(
            sell_pool, sell_pool + buf * 2,
            alpha=0.08, color="#ce93d8", zorder=1,
        )
        ax.axhline(
            sell_pool, color="#ce93d8", linewidth=0.9,
            linestyle=":", zorder=2, alpha=0.7,
        )

        # Label pool di sisi kiri
        ax.text(
            0.2, buy_pool,
            f" 💧 Buy-Side Pool  {buy_pool:.4f}",
            color="#ffd600", fontsize=7.5, va="bottom", alpha=0.85,
        )
        ax.text(
            0.2, sell_pool,
            f" 💧 Sell-Side Pool  {sell_pool:.4f}",
            color="#ce93d8", fontsize=7.5, va="top", alpha=0.85,
        )

    # ═══════════════════════════════════════════════
    # LAYER 1 — TP Zone (hijau)
    # ═══════════════════════════════════════════════
    lo_tp, hi_tp = (entry, tp) if is_buy else (tp, entry)
    ax.axhspan(lo_tp, hi_tp, alpha=0.13, color="#00c853", zorder=1)

    # ═══════════════════════════════════════════════
    # LAYER 2 — SL Zone (merah)  ← anti-sweep
    # ═══════════════════════════════════════════════
    lo_sl, hi_sl = (sl, entry) if is_buy else (entry, sl)
    ax.axhspan(lo_sl, hi_sl, alpha=0.13, color="#d50000", zorder=1)

    # Jika ada sweep_data: gambar "anti-sweep buffer" lebih gelap
    if sweep_data:
        sweep_ref = sweep_data["sweep_ref"]
        if is_buy:
            # buffer zone: sweep_ref s.d. sl  (zona di luar pool → tempat SL kita)
            ax.axhspan(sl, sweep_ref, alpha=0.22, color="#b71c1c", zorder=1)
            ax.axhline(sweep_ref, color="#ff6d00", linewidth=1.0,
                       linestyle="-.", zorder=5, alpha=0.8)
            ax.text(
                xs[-1] + 0.5, sweep_ref,
                f" 🟠 Liq Pool  {sweep_ref:.4f}",
                color="#ff6d00", fontsize=7.5, va="center", fontweight="bold",
            )
        else:
            ax.axhspan(sweep_ref, sl, alpha=0.22, color="#b71c1c", zorder=1)
            ax.axhline(sweep_ref, color="#ff6d00", linewidth=1.0,
                       linestyle="-.", zorder=5, alpha=0.8)
            ax.text(
                xs[-1] + 0.5, sweep_ref,
                f" 🟠 Liq Pool  {sweep_ref:.4f}",
                color="#ff6d00", fontsize=7.5, va="center", fontweight="bold",
            )

    # ═══════════════════════════════════════════════
    # LAYER 3 — Candles
    # ═══════════════════════════════════════════════
    w = 0.6
    for i, x in enumerate(xs):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        col = "#26a69a" if c >= o else "#ef5350"
        ax.plot([x, x], [l, h], color=col, linewidth=0.8, zorder=6)
        body_lo = min(o, c)
        body_hi = max(o, c)
        rect = mpatches.FancyBboxPatch(
            (x - w / 2, body_lo),
            w, max(body_hi - body_lo, (h - l) * 0.003),
            boxstyle="square,pad=0",
            linewidth=0, facecolor=col, zorder=7,
        )
        ax.add_patch(rect)

    # ═══════════════════════════════════════════════
    # LAYER 4 — EMA
    # ═══════════════════════════════════════════════
    ax.plot(xs, df_c["EMA8"].values,  color="#00e5ff", linewidth=1.0, label="EMA 8",  zorder=8)
    ax.plot(xs, df_c["EMA21"].values, color="#ffeb3b", linewidth=1.0, label="EMA 21", zorder=8)
    ax.plot(xs, df_c["EMA50"].values, color="#ff9800", linewidth=1.2, label="EMA 50", zorder=8)

    # ═══════════════════════════════════════════════
    # LAYER 5 — TP / Entry / SL lines + labels
    # ═══════════════════════════════════════════════
    lx = xs[-1] + 0.5   # x posisi label (kanan)

    # TP line
    ax.axhline(tp, color="#00e676", linewidth=1.5, linestyle="--", zorder=9)
    ax.text(lx, tp, f" 🎯 TP  {tp:.4f}", color="#00e676",
            fontsize=8.5, va="center", fontweight="bold")

    # Entry line
    ax.axhline(entry, color="#ffffff", linewidth=1.0, linestyle="--", zorder=9, alpha=0.8)
    ax.text(lx, entry, f" 📍 Entry  {entry:.4f}", color="#ffffff",
            fontsize=8.5, va="center", fontweight="bold", alpha=0.9)

    # SL line  (anti-sweep)
    ax.axhline(sl, color="#ff1744", linewidth=1.5, linestyle="--", zorder=9)
    ax.text(lx, sl, f" 🛡️ SL  {sl:.4f}\n   (anti-sweep)", color="#ff1744",
            fontsize=8.5, va="center", fontweight="bold")

    # ═══════════════════════════════════════════════
    # LAYER 6 — RR bracket (panah di sisi kiri)
    # ═══════════════════════════════════════════════
    bx = -0.8   # x posisi bracket
    # TP bracket
    ax.annotate(
        "", xy=(bx, tp), xytext=(bx, entry),
        arrowprops=dict(arrowstyle="<->", color="#00e676", lw=1.2),
    )
    ax.text(bx - 0.3, (entry + tp) / 2, "TP",
            color="#00e676", fontsize=7, ha="right", va="center")
    # SL bracket
    ax.annotate(
        "", xy=(bx, sl), xytext=(bx, entry),
        arrowprops=dict(arrowstyle="<->", color="#ff1744", lw=1.2),
    )
    ax.text(bx - 0.3, (entry + sl) / 2, "SL",
            color="#ff1744", fontsize=7, ha="right", va="center")

    # ═══════════════════════════════════════════════
    # Axes formatting
    # ═══════════════════════════════════════════════
    tick_every  = max(1, len(xs) // 10)
    tick_idx    = xs[::tick_every]
    tick_labels = [df_c.index[i].strftime("%m/%d %H:%M") for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7, color="#aaaaaa")

    # y-range: cukupkan agar semua level (tp, sl, pool) terlihat
    all_levels = [min(lows), max(highs), tp, sl]
    if sweep_data:
        all_levels += [sweep_data["buy_pool"], sweep_data["sell_pool"]]
    y_lo = min(all_levels)
    y_hi = max(all_levels)
    pad  = (y_hi - y_lo) * 0.06
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.set_xlim(-1.5, xs[-1] + 11)

    ax.yaxis.set_tick_params(labelcolor="#aaaaaa", labelsize=8)
    ax.yaxis.tick_right()
    ax.grid(axis="y", color="#1f2937", linewidth=0.5, linestyle="-")
    ax.grid(axis="x", color="#1f2937", linewidth=0.3, linestyle=":")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1f2937")

    # ═══════════════════════════════════════════════
    # Legend (manual — lebih rapi)
    # ═══════════════════════════════════════════════
    legend_items = [
        Line2D([0], [0], color="#00e5ff", lw=1.2, label="EMA 8"),
        Line2D([0], [0], color="#ffeb3b", lw=1.2, label="EMA 21"),
        Line2D([0], [0], color="#ff9800", lw=1.4, label="EMA 50"),
        mpatches.Patch(facecolor="#00c853", alpha=0.4, label="TP Zone"),
        mpatches.Patch(facecolor="#d50000", alpha=0.4, label="SL Zone"),
    ]
    if sweep_data:
        legend_items += [
            Line2D([0], [0], color="#ffd600", lw=1, linestyle=":", label="Buy-Side Pool"),
            Line2D([0], [0], color="#ce93d8", lw=1, linestyle=":", label="Sell-Side Pool"),
            Line2D([0], [0], color="#ff6d00", lw=1, linestyle="-.", label="Liq Sweep Ref"),
            mpatches.Patch(facecolor="#b71c1c", alpha=0.5, label="Anti-Sweep Buffer"),
        ]
    ax.legend(
        handles=legend_items, loc="upper left", fontsize=7.5,
        facecolor="#161b22", edgecolor="#30363d", labelcolor="#cccccc",
        ncol=2, framealpha=0.85,
    )

    # ═══════════════════════════════════════════════
    # Title
    # ═══════════════════════════════════════════════
    rr_val = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
    title_color = "#00e676" if is_buy else "#ff1744"
    fig.suptitle(
        f"{pair}  ·  {timeframe.upper()}  ·  {action_lbl}  @  {entry:.4f}"
        f"   |   R:R  1:{rr_val:.2f}",
        color=title_color, fontsize=13, fontweight="bold", y=0.97,
    )

    # ═══════════════════════════════════════════════
    # Info bar bawah — ringkasan anti-sweep
    # ═══════════════════════════════════════════════
    if sweep_data:
        sweep_txt = (
            f"🛡️ Anti Liquidity Sweep SL  |  "
            f"Pool ref: {sweep_data['sweep_ref']:.4f}  |  "
            f"Buffer: {sweep_data['buf']:.4f}  |  "
            f"SL beyond pool ({sweep_data['sweep_dir']} pool) → {sl:.4f}  |  "
            f"RR 1:{rr_val:.2f}"
        )
    else:
        sweep_txt = f"TP: {tp:.4f}  |  Entry: {entry:.4f}  |  SL: {sl:.4f}  |  RR 1:{rr_val:.2f}"

    ax2.text(
        0.5, 0.5, sweep_txt,
        transform=ax2.transAxes,
        color="#e0e0e0", fontsize=8.5, ha="center", va="center",
        bbox=dict(facecolor="#161b22", edgecolor="#30363d", boxstyle="round,pad=0.4"),
    )

    # ═══════════════════════════════════════════════
    # Watermark
    # ═══════════════════════════════════════════════
    ax.text(
        0.5, 0.5, "AI INSTITUTIONAL BOT",
        transform=ax.transAxes,
        fontsize=30, color="white", alpha=0.03,
        ha="center", va="center", rotation=30, fontweight="bold",
    )

    filename = f"{pair}_candles.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filename

# ==================================================
# UI VIEWS
# ==================================================

# ── Pair selector dropdown ────────────────────────
class PairSelectView(View):
    def __init__(self, timeframe: str = DEFAULT_TF, mode: str = "signal"):
        super().__init__(timeout=60)
        self.timeframe = timeframe
        self.mode      = mode   # "signal" | "chart"

        options = [
            discord.SelectOption(label=p, description=PAIRS[p], emoji="📈")
            for p in PAIRS
        ]
        select = Select(
            placeholder = "Pilih pair...",
            options     = options,
        )
        select.callback = self.on_pair_select
        self.add_item(select)

    async def on_pair_select(self, interaction: discord.Interaction):
        pair = interaction.data["values"][0]
        await interaction.response.defer()
        await send_signal_or_chart(interaction, pair, self.timeframe, self.mode)

# ── Timeframe selector dropdown ───────────────────
class TimeframeSelectView(View):
    def __init__(self, pair: str, mode: str = "signal", channel_id: int = None):
        super().__init__(timeout=60)
        self.pair       = pair
        self.mode       = mode
        self.channel_id = channel_id   # channel tujuan hasil signal

        options = [
            discord.SelectOption(label=tf.upper(), value=tf, description=f"Interval {tf}")
            for tf in TIMEFRAMES
        ]
        select = Select(placeholder="Pilih timeframe...", options=options)
        select.callback = self.on_tf_select
        self.add_item(select)

    async def on_tf_select(self, interaction: discord.Interaction):
        tf = interaction.data["values"][0]
        # acknowledge dulu agar tidak timeout
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(
            f"⏳ Memproses `{self.pair}` `{tf.upper()}`...", ephemeral=True
        )

        # kirim hasil ke channel asli, bukan ephemeral
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
                    create_chart, data["df"], self.pair, data["entry"], data["tp"], data["sl"], tf, data["sweep"]
                )
                await channel.send(
                    f"📊 **{self.pair}** `{tf.upper()}` Candle Chart",
                    file=discord.File(path),
                )
            else:
                await channel.send(embed=embed, view=view)
        except Exception as e:
            await channel.send(f"❌ Error `{self.pair}` `{tf}`: {e}")

# ── Signal result buttons ─────────────────────────
class SignalActionView(View):
    def __init__(self, data: dict):
        super().__init__(timeout=120)
        self.data = data

    @discord.ui.button(label="📊 Candle Chart", style=discord.ButtonStyle.primary)
    async def show_chart(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        d    = self.data
        try:
            path = await run_blocking(create_chart, d["df"], d["pair"], d["entry"], d["tp"], d["sl"], d["timeframe"], d["sweep"])
            await interaction.followup.send(
                f"📊 **{d['pair']}** `{d['timeframe'].upper()}` Candle Chart",
                file=discord.File(path),
            )
        except Exception as e:
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
            await interaction.followup.send(f"❌ Gagal rescan: `{e}`")

    @discord.ui.button(label="📋 Ganti Timeframe", style=discord.ButtonStyle.secondary)
    async def change_tf(self, interaction: discord.Interaction, button: Button):
        view = TimeframeSelectView(
            self.data["pair"],
            mode       = "signal",
            channel_id = interaction.channel_id,
        )
        await interaction.response.send_message(
            "⏱ Pilih timeframe:", view=view, ephemeral=True
        )

    @discord.ui.button(label="🏆 Best Signal Semua Pair", style=discord.ButtonStyle.success)
    async def best_signal(self, interaction: discord.Interaction, button: Button):
        await interaction.response.defer()
        await interaction.followup.send(f"⏳ Scanning semua pair `{self.data['timeframe'].upper()}`...")
        await scan_best_and_send(interaction, self.data["timeframe"])

# ── Main menu buttons ─────────────────────────────
class MainMenuView(View):
    def __init__(self):
        super().__init__(timeout=120)

    @discord.ui.button(label="🏆 Best Signal", style=discord.ButtonStyle.success, row=0)
    async def best_signal(self, interaction: discord.Interaction, button: Button):
        # First show TF selector
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
        await interaction.response.send_message(
            f"Auto Signal sekarang: **{status}**", ephemeral=True
        )

class TimeframeSelectMenu(View):
    """Standalone TF picker untuk best-signal mode."""
    def __init__(self, mode: str = "best"):
        super().__init__(timeout=60)
        self.mode = mode
        options = [
            discord.SelectOption(label=tf.upper(), value=tf)
            for tf in TIMEFRAMES
        ]
        sel = Select(placeholder="Pilih timeframe...", options=options)
        sel.callback = self.on_select
        self.add_item(sel)

    async def on_select(self, interaction: discord.Interaction):
        tf = interaction.data["values"][0]
        await interaction.response.defer(ephemeral=True)
        await interaction.followup.send(
            f"⏳ Mencari best signal `{tf.upper()}`...", ephemeral=True
        )
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
        path  = await run_blocking(create_chart, best["df"], best["pair"], best["entry"], best["tp"], best["sl"], best["timeframe"], best["sweep"])
        embed = build_embed(best)
        view  = SignalActionView(best)
        await channel.send(embed=embed, view=view, file=discord.File(path))

# ==================================================
# HELPER: send signal or chart
# ==================================================
async def send_signal_or_chart(
    interaction: discord.Interaction,
    pair: str,
    timeframe: str,
    mode: str,
):
    try:
        # jalankan di thread pool agar tidak freeze event loop
        data = await run_blocking(generate_signal, pair, timeframe)

        if mode == "chart":
            path = await run_blocking(
                create_chart, data["df"], pair, data["entry"], data["tp"], data["sl"], data["timeframe"], data["sweep"]
            )
            await interaction.followup.send(
                f"📊 **{pair}** `{timeframe.upper()}` Candle Chart",
                file=discord.File(path),
            )
        else:
            embed = build_embed(data)
            view  = SignalActionView(data)
            await interaction.followup.send(embed=embed, view=view)

    except Exception as e:
        print(f"send_signal_or_chart ERROR [{pair}|{timeframe}]: {e}")
        await interaction.followup.send(f"❌ Error memproses {pair}: `{e}`")

# ==================================================
# HELPER: scan best pair
# ==================================================
async def scan_best_and_send(interaction: discord.Interaction, timeframe: str):
    # scan semua pair secara concurrent (bukan sequential)
    async def _safe_signal(pair):
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

    best = max(results, key=lambda x: x["score"])
    path  = await run_blocking(create_chart, best["df"], best["pair"], best["entry"], best["tp"], best["sl"], best["timeframe"], best["sweep"])
    embed = build_embed(best)
    view  = SignalActionView(best)
    await interaction.followup.send(embed=embed, view=view, file=discord.File(path))

# ==================================================
# HELPER: scan all pairs
# ==================================================
async def scan_all_pairs_and_send(interaction: discord.Interaction, timeframe: str):
    async def _safe_signal(pair):
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
            lines.append(
                f"{emoji} **{pair}** | Score: `{d['score']}%` | {d['confidence']}"
                f" | Entry: `{d['entry']:.4f}`"
            )

    embed = discord.Embed(
        title       = f"🔁 Scan Semua Pair — {timeframe.upper()}",
        description = "\n".join(lines),
        color       = discord.Color.blurple(),
    )
    embed.set_footer(text="Klik '📈 Signal per Pair' untuk detail sinyal per pair")
    await interaction.followup.send(embed=embed)

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
    embed.add_field(name="🏆 Best Signal",       value="Sinyal terbaik dari semua pair",          inline=False)
    embed.add_field(name="📈 Signal per Pair",   value="Sinyal spesifik untuk satu pair",         inline=False)
    embed.add_field(name="📊 Chart per Pair",    value="Candle chart tanpa sinyal",                inline=False)
    embed.add_field(name="🔁 Scan Semua Pair",   value="Ringkasan cepat semua pair",              inline=False)
    embed.add_field(name="⚙️ Auto Signal Toggle", value="Aktifkan/matikan auto-signal otomatis",  inline=False)
    await interaction.response.send_message(embed=embed, view=MainMenuView())

@client.tree.command(name="signal", description="Best AI signal dari semua pair")
@app_commands.describe(timeframe="Timeframe: 5m / 15m / 1h / 4h")
async def signal_cmd(interaction: discord.Interaction, timeframe: str = DEFAULT_TF):
    if timeframe not in TIMEFRAMES:
        await interaction.response.send_message(
            f"❌ Timeframe tidak valid. Pilih: {', '.join(TIMEFRAMES.keys())}", ephemeral=True
        )
        return
    await interaction.response.defer()
    await interaction.followup.send(f"⏳ Scanning semua pair `{timeframe.upper()}`...", ephemeral=False)
    await scan_best_and_send(interaction, timeframe)

@client.tree.command(name="pair", description="Sinyal AI untuk pair tertentu")
@app_commands.describe(pair="Contoh: BTCUSD", timeframe="Timeframe: 5m / 15m / 1h / 4h")
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
@app_commands.describe(pair="Contoh: ETHUSD", timeframe="Timeframe: 5m / 15m / 1h / 4h")
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
@app_commands.describe(timeframe="Timeframe: 5m / 15m / 1h / 4h")
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

# ==================================================
# AUTO SIGNAL LOOP
# ==================================================
async def auto_signal_loop():
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)

    while not client.is_closed():
        try:
            if AUTO_SIGNAL and channel:
                for pair in PAIRS:
                    try:
                        data = await run_blocking(generate_signal, pair, DEFAULT_TF)
                        if data["score"] >= AUTO_MIN_SCORE:
                            path  = await run_blocking(create_chart, data["df"], pair, data["entry"], data["tp"], data["sl"], data["timeframe"], data["sweep"])
                            embed = build_embed(data)
                            view  = SignalActionView(data)
                            await channel.send(embed=embed, view=view, file=discord.File(path))
                    except Exception as e:
                        print(f"AUTO ERROR {pair}: {e}")

            await asyncio.sleep(900)   # setiap 15 menit
        except Exception as e:
            print(f"LOOP ERROR: {e}")
            await asyncio.sleep(30)

# ==================================================
# READY
# ==================================================
@client.event
async def on_ready():
    await client.tree.sync()
    print(f"✅ AI TRADING BOT READY — Logged in as {client.user}")

# ==================================================
# START
# ==================================================
async def main():
    async with client:
        client.loop.create_task(auto_signal_loop())
        await client.start(TOKEN)

asyncio.run(main())
