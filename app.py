import discord
from discord import app_commands
import asyncio
import os
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# ENV
# =========================
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

PAIRS = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XRPUSD": "XRP-USD",
    "SOLUSD": "SOL-USD",
    "XAUUSD": "GC=F"
}

TIMEFRAME = "5m"
AUTO_SIGNAL = False

# =========================
# DISCORD CLIENT
# =========================
class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

client = Bot()

# =========================
# DATA
# =========================
def get_data(symbol):
    df = yf.download(symbol, period="5d", interval="15m")
    df = df.dropna()
    return df

# =========================
# INDICATORS
# =========================
def ema(s, p):
    return s.ewm(span=p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = -d.clip(upper=0).rolling(p).mean()
    rs = g / l
    return 100 - (100 / (1 + rs))

def macd(s):
    m1 = ema(s, 12)
    m2 = ema(s, 26)
    macd = m1 - m2
    sig = ema(macd, 9)
    return macd, sig

# =========================
# SMC
# =========================
def smc(df):
    last = df.iloc[-1]

    bos = last["Close"] > df["High"].rolling(10).max().iloc[-2]
    choch = last["Close"] < df["Low"].rolling(10).min().iloc[-2]
    liq = last["High"] > df["High"].rolling(20).max().iloc[-2]

    return {"BOS": bool(bos), "CHoCH": bool(choch), "LIQ": bool(liq)}

# =========================
# LIQUIDITY
# =========================
def liquidity(df):
    high = df["High"].rolling(20).max().iloc[-1]
    low = df["Low"].rolling(20).min().iloc[-1]
    price = df["Close"].iloc[-1]

    return {
        "HIGH": high,
        "LOW": low,
        "PRESSURE": "BUY" if abs(price-low) < abs(price-high) else "SELL"
    }

# =========================
# FLOW
# =========================
def flow(df):
    momentum = df["Close"].diff().mean()
    return {
        "FLOW": "BUY" if momentum > 0 else "SELL",
        "STRENGTH": abs(momentum) * 100
    }

# =========================
# LSTM (SIMPLIFIED)
# =========================
def lstm(df):
    return "BULLISH" if df["Close"].iloc[-1] > df["Close"].mean() else "BEARISH"

# =========================
# AI SCORE
# =========================
def score(l, s, liq, fl):
    sc = 50

    if l == "BULLISH": sc += 20
    else: sc -= 20

    if s["BOS"]: sc += 10
    if s["LIQ"]: sc += 10
    if s["CHoCH"]: sc -= 10

    if liq["PRESSURE"] == "BUY": sc += 10
    else: sc -= 10

    if fl["FLOW"] == "BUY": sc += 10
    else: sc -= 10

    return max(0, min(100, sc))

# =========================
# GENERATE SIGNAL
# =========================
def generate(pair):
    df = get_data(PAIRS[pair])
    close = df["Close"]

    r = rsi(close).iloc[-1]
    m1, m2 = macd(close)
    m = m1.iloc[-1] - m2.iloc[-1]

    s = smc(df)
    liq = liquidity(df)
    fl = flow(df)
    l = lstm(df)

    sc = score(l, s, liq, fl)

    price = float(close.iloc[-1])

    return {
        "pair": pair,
        "score": sc,
        "lstm": l,
        "rsi": r,
        "macd": m,
        "smc": s,
        "liq": liq,
        "flow": fl,
        "entry": price,
        "tp": price + (price * 0.01),
        "sl": price - (price * 0.01)
    }

# =========================
# FORMAT
# =========================
def fmt(d):
    return f"""
📊 {d['pair']} AI INSTITUTIONAL SIGNAL

🧠 SCORE: {d['score']}%

📈 LSTM: {d['lstm']}
📊 RSI: {d['rsi']:.2f}
📊 MACD: {d['macd']:.2f}

🏗 BOS: {d['smc']['BOS']}
🔄 CHoCH: {d['smc']['CHoCH']}
💧 LIQ: {d['smc']['LIQ']}

💧 Pressure: {d['liq']['PRESSURE']}
🏦 Flow: {d['flow']['FLOW']} ({d['flow']['STRENGTH']:.2f})

🟢 ENTRY: {d['entry']}
🎯 TP: {d['tp']}
🛑 SL: {d['sl']}
"""

# =========================
# CHART 5M ZONE
# =========================
def chart(df, pair, entry, tp, sl):
    plt.figure(figsize=(10,4))
    plt.plot(df["Close"], label="Price")

    plt.axhline(entry, linestyle="--", color="blue")
    plt.axhline(tp, linestyle="--", color="green")
    plt.axhline(sl, linestyle="--", color="red")

    plt.title(f"{pair} 5M ENTRY ZONE")

    file = f"{pair}_signal.png"
    plt.savefig(file)
    plt.close()
    return file

# =========================
# BEST PAIR SCANNER (/signal FIX)
# =========================
@client.tree.command(name="signal")
async def signal(interaction: discord.Interaction):

    await interaction.response.defer()

    best = None
    best_score = -999
    best_df = None

    for p in PAIRS:
        d = generate(p)

        if d["score"] > best_score:
            best = d
            best_score = d["score"]
            best_df = get_data(PAIRS[p])

    file = chart(best_df, best["pair"], best["entry"], best["tp"], best["sl"])

    await interaction.followup.send(
        fmt(best),
        file=discord.File(file)
    )

# =========================
# SHOW CHART ONLY
# =========================
@client.tree.command(name="show_trend_chart")
async def chart_cmd(interaction: discord.Interaction, pair: str):
    df = get_data(PAIRS[pair])
    file = chart(df, pair, 0, 0, 0)
    await interaction.response.send_message(file=discord.File(file))

# =========================
# AUTO TOGGLE
# =========================
@client.tree.command(name="auto_toggle")
async def toggle(interaction: discord.Interaction):
    global AUTO_SIGNAL
    AUTO_SIGNAL = not AUTO_SIGNAL
    await interaction.response.send_message(f"AUTO: {AUTO_SIGNAL}")

# =========================
# AUTO LOOP
# =========================
async def loop():
    await client.wait_until_ready()
    ch = client.get_channel(CHANNEL_ID)

    while True:
        if AUTO_SIGNAL:
            for p in PAIRS:
                d = generate(p)
                if d["score"] >= 75:
                    df = get_data(PAIRS[p])
                    file = chart(df, p, d["entry"], d["tp"], d["sl"])
                    await ch.send(fmt(d), file=discord.File(file))

        await asyncio.sleep(900)

# =========================
# READY
# =========================
@client.event
async def on_ready():
    await client.tree.sync()
    print("V4 AI TRADING BOT READY")

asyncio.get_event_loop().create_task(loop())

client.run(TOKEN)
