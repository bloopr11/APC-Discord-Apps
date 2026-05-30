# ==================================================
# price_alert.py  —  Price Change Alert System
#
# Deteksi perubahan harga > threshold% dalam window
# waktu tertentu, kirim notifikasi ke Discord.
#
# ENV (opsional, ada default):
#   PRICE_ALERT_THRESHOLD = 5.0   (persen, default 5%)
#   PRICE_ALERT_WINDOW    = 60    (menit lookback, default 60)
#   PRICE_ALERT_COOLDOWN  = 120   (menit cooldown per pair, default 120)
#   PRICE_ALERT_INTERVAL  = 5     (menit antar scan, default 5)
# ==================================================

import os
import time
import asyncio
import datetime as dt
import discord

# ==================================================
# CONFIG
# ==================================================
THRESHOLD    = float(os.getenv("PRICE_ALERT_THRESHOLD", "5.0"))   # %
WINDOW_MIN   = int(os.getenv("PRICE_ALERT_WINDOW",    "60"))      # menit lookback
COOLDOWN_MIN = int(os.getenv("PRICE_ALERT_COOLDOWN",  "120"))     # menit cooldown
SCAN_MIN     = int(os.getenv("PRICE_ALERT_INTERVAL",  "5"))       # menit antar scan

# State: simpan harga referensi & last alert per pair
# format: { "BTCUSD": {"ref_price": float, "ref_ts": float, "last_alert_ts": float} }
_price_state: dict = {}


# ==================================================
# CORE DETECTION
# ==================================================

def check_price_change(pair: str, df) -> dict | None:
    """
    Cek apakah ada perubahan harga > THRESHOLD% dalam WINDOW_MIN menit.
    
    Return dict jika ada alert, None jika tidak.
    """
    import pandas as pd

    if df is None or len(df) < 2:
        return None

    now_price = float(df["Close"].iloc[-1])
    now_ts    = time.time()

    # Ambil harga WINDOW_MIN menit yang lalu dari data
    window_td = dt.timedelta(minutes=WINDOW_MIN)
    cutoff    = df.index[-1] - window_td

    # Cari candle paling dekat dengan cutoff
    past_df = df[df.index >= cutoff]
    if len(past_df) < 2:
        past_df = df.iloc[-max(2, min(12, len(df))):]

    ref_price = float(past_df["Close"].iloc[0])
    ref_ts    = past_df.index[0]

    if ref_price == 0:
        return None

    pct_change = ((now_price - ref_price) / ref_price) * 100

    if abs(pct_change) < THRESHOLD:
        return None

    # Cek cooldown
    state     = _price_state.get(pair, {})
    last_alert = state.get("last_alert_ts", 0)
    if (now_ts - last_alert) < COOLDOWN_MIN * 60:
        return None

    direction = "BULLISH" if pct_change > 0 else "BEARISH"

    # Hitung high/low dalam window
    high_in_window = float(past_df["High"].max())
    low_in_window  = float(past_df["Low"].min())

    # Volume spike check
    vol_spike = False
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        avg_vol   = float(df["Volume"].iloc[-20:].mean())
        last_vol  = float(df["Volume"].iloc[-1])
        vol_spike = last_vol > avg_vol * 1.5

    return {
        "pair":        pair,
        "direction":   direction,
        "pct_change":  round(pct_change, 2),
        "now_price":   now_price,
        "ref_price":   ref_price,
        "ref_time":    ref_ts,
        "window_min":  WINDOW_MIN,
        "high":        high_in_window,
        "low":         low_in_window,
        "vol_spike":   vol_spike,
        "threshold":   THRESHOLD,
    }


# ==================================================
# EMBED BUILDER
# ==================================================

def build_price_alert_embed(alert: dict) -> discord.Embed:
    is_bull    = alert["direction"] == "BULLISH"
    color      = discord.Color.green() if is_bull else discord.Color.red()
    arrow      = "🚀" if is_bull else "💥"
    pct        = alert["pct_change"]
    pair       = alert["pair"]
    now_price  = alert["now_price"]
    ref_price  = alert["ref_price"]
    window     = alert["window_min"]

    # Strength label
    abs_pct = abs(pct)
    if abs_pct >= 15:   strength = "🔥 EXTREME"
    elif abs_pct >= 10: strength = "⚡ VERY STRONG"
    elif abs_pct >= 7:  strength = "💪 STRONG"
    else:               strength = "📊 SIGNIFICANT"

    embed = discord.Embed(
        title = (
            f"{arrow} PRICE ALERT — {pair}  "
            f"[{'+' if is_bull else ''}{pct:.2f}%]"
        ),
        description = (
            f"**{strength}** {'Bullish' if is_bull else 'Bearish'} move "
            f"dalam `{window}` menit terakhir\n"
            f"Threshold: `{alert['threshold']}%`"
        ),
        color = color,
    )

    embed.add_field(
        name  = "💰 Harga",
        value = (
            f"Sekarang: `{now_price:.4f}`\n"
            f"{window}m lalu:  `{ref_price:.4f}`\n"
            f"Perubahan: `{'+' if is_bull else ''}{pct:.2f}%`"
        ),
        inline=True,
    )

    embed.add_field(
        name  = f"📊 Range {window}m",
        value = (
            f"High: `{alert['high']:.4f}`\n"
            f"Low:  `{alert['low']:.4f}`\n"
            f"Vol spike: `{'YES ⚡' if alert['vol_spike'] else 'NO'}`"
        ),
        inline=True,
    )

    # Saran tindakan
    if is_bull:
        action_hint = (
            "• Cek apakah ada **BOS Bullish** di timeframe lebih tinggi\n"
            "• Tunggu **pullback ke FVG/OB** untuk entry\n"
            "• Waspadai **overbought** jika RSI > 70"
        )
    else:
        action_hint = (
            "• Cek apakah ada **BOS Bearish** di timeframe lebih tinggi\n"
            "• Tunggu **retest ke FVG/OB** untuk entry SELL\n"
            "• Waspadai **oversold** jika RSI < 30"
        )

    embed.add_field(
        name  = "💡 Saran",
        value = action_hint,
        inline=False,
    )

    ref_time_str = alert["ref_time"].strftime("%H:%M UTC") if hasattr(alert["ref_time"], "strftime") else str(alert["ref_time"])
    embed.set_footer(
        text=f"Ref time: {ref_time_str} | Cooldown: {COOLDOWN_MIN}m | APC Engine"
    )
    embed.timestamp = discord.utils.utcnow()

    return embed


# ==================================================
# MAIN LOOP
# ==================================================

async def price_alert_loop(client: discord.Client):
    """
    Background loop — scan semua pair setiap SCAN_MIN menit.
    Dipanggil dari main() di app.py.
    """
    await client.wait_until_ready()
    await asyncio.sleep(60)   # tunggu 1 menit setelah bot ready

    print(f"[PRICE ALERT] Loop started — threshold={THRESHOLD}% window={WINDOW_MIN}m")

    # Import dari app.py (sudah di-load saat module ini dipakai)
    from app import PAIRS, CHANNEL_ID, get_data, DEFAULT_TF
    import functools

    channel = client.get_channel(CHANNEL_ID)

    while not client.is_closed():
        try:
            if channel:
                for pair in PAIRS:
                    try:
                        # Ambil data 15m untuk deteksi perubahan
                        loop = asyncio.get_event_loop()
                        df   = await loop.run_in_executor(
                            None, functools.partial(get_data, pair, "15m")
                        )

                        alert = check_price_change(pair, df)

                        if alert:
                            embed   = build_price_alert_embed(alert)
                            is_bull = alert["direction"] == "BULLISH"
                            pct     = alert["pct_change"]

                            await channel.send(
                                content = (
                                    f"{'🚀' if is_bull else '💥'} **PRICE ALERT** — "
                                    f"`{pair}` bergerak `{'+' if is_bull else ''}{pct:.2f}%` "
                                    f"dalam `{alert['window_min']}` menit!"
                                ),
                                embed = embed,
                            )

                            # Update state (cooldown)
                            _price_state[pair] = {
                                "last_alert_ts": time.time(),
                                "last_pct":      pct,
                                "last_dir":      alert["direction"],
                            }

                            print(
                                f"[PRICE ALERT] {pair} {'+' if is_bull else ''}{pct:.2f}% "
                                f"— alert sent"
                            )

                        await asyncio.sleep(0.5)   # jeda antar pair

                    except Exception as e:
                        print(f"[PRICE ALERT ERROR] {pair}: {e}")

        except Exception as e:
            print(f"[PRICE ALERT LOOP ERROR] {e}")

        await asyncio.sleep(SCAN_MIN * 60)


# ==================================================
# SLASH COMMAND HELPER
# (tambahkan ke app.py jika ingin /alertstatus)
# ==================================================

def build_alertstatus_embed() -> discord.Embed:
    """Tampilkan status alert terakhir semua pair."""
    now   = time.time()
    lines = []

    for pair, state in _price_state.items():
        elapsed = int(now - state.get("last_alert_ts", 0))
        remain  = max(0, COOLDOWN_MIN * 60 - elapsed)
        pct     = state.get("last_pct", 0)
        dirn    = state.get("last_dir", "?")
        emoji   = "🚀" if dirn == "BULLISH" else "💥"
        lines.append(
            f"{emoji} **{pair}** `{'+' if pct > 0 else ''}{pct:.2f}%` — "
            f"{elapsed//60}m lalu | cooldown sisa {remain//60}m"
        )

    if not lines:
        lines = ["Belum ada price alert sejak bot start."]

    embed = discord.Embed(
        title       = "📊 Price Alert Status",
        description = "\n".join(lines),
        color       = discord.Color.blurple(),
    )
    embed.set_footer(
        text=f"Threshold: {THRESHOLD}% | Window: {WINDOW_MIN}m | Cooldown: {COOLDOWN_MIN}m"
    )
    return embed