import discord
from discord import app_commands
import asyncio
import os
import pandas as pd
import numpy as np
import yfinance as yf

import matplotlib
matplotlib.use("Agg")

import mplfinance as mpf

# ==================================================
# ENV
# ==================================================
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

# ==================================================
# PAIRS
# ==================================================
PAIRS = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XAUUSD": "GC=F",
    "XRPUSD": "XRP-USD",
    "SOLUSD": "SOL-USD",
    "ADAUSD": "ADA-USD",
    "DOGEUSD": "DOGE-USD"
}

TIMEFRAME = "5m"
AUTO_SIGNAL = False

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
# GET MARKET DATA
# ==================================================
def get_data(symbol):

    df = yf.download(
        symbol,
        period="5d",
        interval="5m",
        auto_adjust=True,
        progress=False
    )

    df = df.dropna()

    # FIX multi-index columns
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    return df

# ==================================================
# EMA
# ==================================================
def ema(series, period):
    return series.ewm(span=period).mean()

# ==================================================
# RSI
# ==================================================
def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))

# ==================================================
# MACD
# ==================================================
def macd(series):

    ema12 = ema(series, 12)
    ema26 = ema(series, 26)

    macd_line = ema12 - ema26
    signal_line = ema(macd_line, 9)

    return macd_line, signal_line

# ==================================================
# SMC ENGINE
# ==================================================
def smc(df):

    bos = (
        df["Close"].iloc[-1]
        > df["High"].rolling(10).max().iloc[-2]
    )

    choch = (
        df["Close"].iloc[-1]
        < df["Low"].rolling(10).min().iloc[-2]
    )

    liquidity_sweep = (
        df["High"].iloc[-1]
        > df["High"].rolling(20).max().iloc[-2]
    )

    return {
        "BOS": bos,
        "CHoCH": choch,
        "LIQ": liquidity_sweep
    }

# ==================================================
# LIQUIDITY ENGINE
# ==================================================
def liquidity(df):

    high_zone = df["High"].rolling(20).max().iloc[-1]
    low_zone = df["Low"].rolling(20).min().iloc[-1]

    current_price = df["Close"].iloc[-1]

    pressure = (
        "BUY"
        if abs(current_price - low_zone)
        < abs(current_price - high_zone)
        else "SELL"
    )

    return {
        "HIGH_ZONE": high_zone,
        "LOW_ZONE": low_zone,
        "PRESSURE": pressure
    }

# ==================================================
# FLOW ENGINE
# ==================================================
def flow(df):

    momentum = df["Close"].diff().mean()

    return {
        "FLOW": "BUY" if momentum > 0 else "SELL",
        "STRENGTH": abs(momentum) * 100
    }

# ==================================================
# LSTM PROXY
# ==================================================
def lstm_prediction(df):

    current_price = df["Close"].iloc[-1]
    mean_price = df["Close"].mean()

    return (
        "BULLISH"
        if current_price > mean_price
        else "BEARISH"
    )

# ==================================================
# AI SCORE
# ==================================================
def ai_score(lstm, smc_data, liq, fl):

    score = 50

    # LSTM
    if lstm == "BULLISH":
        score += 20
    else:
        score -= 20

    # SMC
    if smc_data["BOS"]:
        score += 10

    if smc_data["LIQ"]:
        score += 10

    if smc_data["CHoCH"]:
        score -= 10

    # Liquidity
    if liq["PRESSURE"] == "BUY":
        score += 10
    else:
        score -= 10

    # Flow
    if fl["FLOW"] == "BUY":
        score += 10
    else:
        score -= 10

    return max(0, min(100, score))

# ==================================================
# GENERATE SIGNAL
# ==================================================
def generate_signal(pair):

    df = get_data(PAIRS[pair])

    close = df["Close"]

    # indicators
    rsi_value = rsi(close).iloc[-1]

    macd_line, macd_signal = macd(close)

    macd_value = (
        macd_line.iloc[-1]
        - macd_signal.iloc[-1]
    )

    # AI
    smc_data = smc(df)
    liq = liquidity(df)
    fl = flow(df)
    lstm = lstm_prediction(df)

    score = ai_score(
        lstm,
        smc_data,
        liq,
        fl
    )

    action = (
        "BUY"
        if score >= 55
        else "SELL"
    )

    price = float(close.iloc[-1])

    if action == "BUY":

        tp = price + (price * 0.01)
        sl = price - (price * 0.01)

    else:

        tp = price - (price * 0.01)
        sl = price + (price * 0.01)

    return {
        "pair": pair,
        "score": score,
        "action": action,
        "entry": price,
        "tp": tp,
        "sl": sl,
        "rsi": rsi_value,
        "macd": macd_value,
        "lstm": lstm,
        "smc": smc_data,
        "liq": liq,
        "flow": fl,
        "df": df
    }

# ==================================================
# FORMAT SIGNAL
# ==================================================
def format_signal(data):

    return f"""
📊 {data['pair']} AI INSTITUTIONAL SIGNAL

🧠 AI SCORE: {data['score']}%

📈 LSTM: {data['lstm']}
📊 RSI: {data['rsi']:.2f}
📊 MACD: {data['macd']:.2f}

🏗 BOS: {data['smc']['BOS']}
🔄 CHoCH: {data['smc']['CHoCH']}
💧 LIQ: {data['smc']['LIQ']}

💧 Pressure: {data['liq']['PRESSURE']}
🏦 Flow: {data['flow']['FLOW']}

🟢 ACTION: {data['action']}

📍 ENTRY: {data['entry']:.4f}
🎯 TP: {data['tp']:.4f}
🛑 SL: {data['sl']:.4f}

⏱ TF: {TIMEFRAME}
"""

# ==================================================
# CREATE CANDLE CHART
# ==================================================
def create_chart(df, pair, entry, tp, sl):

    df_chart = df.copy()

    df_chart.index.name = "Date"

    # EMA
    df_chart["EMA20"] = ema(df_chart["Close"], 20)
    df_chart["EMA50"] = ema(df_chart["Close"], 50)

    addplots = [

        mpf.make_addplot(
            df_chart["EMA20"]
        ),

        mpf.make_addplot(
            df_chart["EMA50"]
        ),

        mpf.make_addplot(
            [entry] * len(df_chart),
            linestyle="--"
        ),

        mpf.make_addplot(
            [tp] * len(df_chart),
            linestyle="--"
        ),

        mpf.make_addplot(
            [sl] * len(df_chart),
            linestyle="--"
        )
    ]

    filename = f"{pair}_candles.png"

    mpf.plot(
        df_chart,
        type="candle",
        style="charles",
        volume=False,
        title=f"{pair} 5M SIGNAL",
        addplot=addplots,
        savefig=filename
    )

    return filename

# ==================================================
# /SIGNAL
# ==================================================
@client.tree.command(
    name="signal",
    description="Best AI signal scanner"
)
async def signal(interaction: discord.Interaction):

    await interaction.response.defer()

    best_signal = None
    best_score = -999

    for pair in PAIRS.keys():

        try:

            signal_data = generate_signal(pair)

            if signal_data["score"] > best_score:

                best_score = signal_data["score"]
                best_signal = signal_data

        except Exception as e:

            print(f"ERROR {pair}: {e}")

    if best_signal is None:

        await interaction.followup.send(
            "❌ Tidak ada signal tersedia."
        )
        return

    chart_file = create_chart(
        best_signal["df"],
        best_signal["pair"],
        best_signal["entry"],
        best_signal["tp"],
        best_signal["sl"]
    )

    await interaction.followup.send(
        format_signal(best_signal),
        file=discord.File(chart_file)
    )

# ==================================================
# /SHOW_TREND_CHART
# ==================================================
@client.tree.command(
    name="show_trend_chart",
    description="Show trend chart pair"
)
async def show_chart(
    interaction: discord.Interaction,
    pair: str
):

    pair = pair.upper()

    if pair not in PAIRS:

        await interaction.response.send_message(
            f"""
❌ Pair tidak tersedia.

Available:
{', '.join(PAIRS.keys())}
"""
        )
        return

    await interaction.response.defer()

    signal_data = generate_signal(pair)

    chart_file = create_chart(
        signal_data["df"],
        pair,
        signal_data["entry"],
        signal_data["tp"],
        signal_data["sl"]
    )

    await interaction.followup.send(
        f"📊 {pair} 5M Candle Chart",
        file=discord.File(chart_file)
    )

# ==================================================
# /AUTO_TOGGLE
# ==================================================
@client.tree.command(
    name="auto_toggle",
    description="Toggle auto signal"
)
async def auto_toggle(interaction: discord.Interaction):

    global AUTO_SIGNAL

    AUTO_SIGNAL = not AUTO_SIGNAL

    await interaction.response.send_message(
        f"AUTO SIGNAL: {AUTO_SIGNAL}"
    )

# ==================================================
# AUTO SIGNAL LOOP
# ==================================================
async def auto_signal_loop():

    await client.wait_until_ready()

    channel = client.get_channel(CHANNEL_ID)

    while not client.is_closed():

        try:

            if AUTO_SIGNAL:

                for pair in PAIRS.keys():

                    try:

                        signal_data = generate_signal(pair)

                        if signal_data["score"] >= 75:

                            chart_file = create_chart(
                                signal_data["df"],
                                pair,
                                signal_data["entry"],
                                signal_data["tp"],
                                signal_data["sl"]
                            )

                            await channel.send(
                                format_signal(signal_data),
                                file=discord.File(chart_file)
                            )

                    except Exception as e:

                        print(f"AUTO ERROR {pair}: {e}")

            await asyncio.sleep(900)

        except Exception as e:

            print(f"LOOP ERROR: {e}")

            await asyncio.sleep(30)

# ==================================================
# READY
# ==================================================
@client.event
async def on_ready():

    await client.tree.sync()

    print("V6 INSTITUTIONAL AI BOT READY")

# ==================================================
# START
# ==================================================
async def main():

    async with client:

        client.loop.create_task(
            auto_signal_loop()
        )

        await client.start(TOKEN)

asyncio.run(main())
