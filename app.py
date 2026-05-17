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
# ENV CONFIG
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

TIMEFRAME = "15m"
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
# MARKET DATA (TRADINGVIEW STYLE)
# =========================
def get_data(symbol):
    df = yf.download(symbol, period="5d", interval="15m")
    df = df.dropna()
    return df

# =========================
# INDICATORS (EMA RSI MACD)
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
# SMC ENGINE (REAL STRUCTURE)
# =========================
def smc(df):
    last = df.iloc[-1]

    bos = last["Close"] > df["High"].rolling(10).max().iloc[-2]
    choch = last["Close"] < df["Low"].rolling(10).min().iloc[-2]
    liquidity = last["High"] > df["High"].rolling(20).max().iloc[-2]

    return {
        "BOS": bool(bos),
        "CHoCH": bool(choch),
        "LIQ": bool(liquidity)
    }

# =========================
# ORDER BLOCK DETECTION
# =========================
def order_block(df):
    last = df.iloc[-1]

    bullish_ob = last["Close"] > df["High"].rolling(5).max().iloc[-2]
    bearish_ob = last["Close"] < df["Low"].rolling(5).min().iloc[-2]

    return {
        "BULLISH_OB": bool(bullish_ob),
        "BEARISH_OB": bool(bearish_ob)
    }

# =========================
# LIQUIDITY HEATMAP
# =========================
def liquidity(df):
    high_zone = df["High"].rolling(20).max().iloc[-1]
    low_zone = df["Low"].rolling(20).min().iloc[-1]

    price = df["Close"].iloc[-1]

    return {
        "HIGH_ZONE": high_zone,
        "LOW_ZONE": low_zone,
        "PRESSURE": "BUY" if abs(price-low_zone) < abs(price-high_zone) else "SELL"
    }

# =========================
# INSTITUTIONAL FLOW
# =========================
def flow(df):
    momentum = df["Close"].diff().mean()

    return {
        "FLOW": "BUY" if momentum > 0 else "SELL",
        "STRENGTH": abs(momentum)*100
    }

# =========================
# LSTM SIMPLIFIED PREDICTION (NO TRAIN MODEL - SAFE VERSION)
# =========================
def lstm_signal(df):
    close = df["Close"]
    future = close.mean() + (close.diff().mean() * 3)

    return "BULLISH" if future > close.iloc[-1] else "BEARISH"

# =========================
# AI SCORE ENGINE
# =========================
def ai_score(lstm, smc, liq, flow):
    score = 50

    if lstm == "BULLISH":
        score += 20
    else:
        score -= 20

    if smc["BOS"]:
        score += 10
    if smc["LIQ"]:
        score += 10
    if smc["CHoCH"]:
        score -= 10

    if liq["PRESSURE"] == "BUY":
        score += 10
    else:
        score -= 10

    if flow["FLOW"] == "BUY":
        score += 10
    else:
        score -= 10

    return max(0, min(100, score))

# =========================
# SIGNAL ENGINE
# =========================
def generate(pair):
    df = get_data(PAIRS[pair])
    close = df["Close"]

    r = rsi(close).iloc[-1]
    macd_line, macd_sig = macd(close)
    m = macd_line.iloc[-1] - macd_sig.iloc[-1]

    s = smc(df)
    ob = order_block(df)
    liq = liquidity(df)
    fl = flow(df)
    lstm = lstm_signal(df)

    score = ai_score(lstm, s, liq, fl)

    action = "BUY" if score >= 55 else "SELL"

    price = float(close.iloc[-1])

    return {
        "pair": pair,
        "score": score,
        "lstm": lstm,
        "rsi": round(r,2),
        "macd": round(m,2),
        "smc": s,
        "ob": ob,
        "liq": liq,
        "flow": fl,
        "action": action,
        "entry": price,
        "tp": price + 100,
        "sl": price - 100
    }

# =========================
# FORMAT MESSAGE
# =========================
def fmt(d):
    return f"""
📊 {d['pair']} INSTITUTIONAL AI SIGNAL

🧠 AI SCORE: {d['score']}%

📈 LSTM: {d['lstm']}
📊 RSI: {d['rsi']}
📊 MACD: {d['macd']}

🏗 BOS: {d['smc']['BOS']}
🔄 CHoCH: {d['smc']['CHoCH']}
💧 LIQ: {d['smc']['LIQ']}

📦 Order Block: {d['ob']}
💧 Liquidity Pressure: {d['liq']['PRESSURE']}
🏦 Flow: {d['flow']['FLOW']} ({d['flow']['STRENGTH']:.2f})

🟢 ACTION: {d['action']}
Entry: {d['entry']}
TP: {d['tp']}
SL: {d['sl']}
"""

# =========================
# CHART
# =========================
def chart(df, pair):
    plt.figure(figsize=(10,4))
    plt.plot(df["Close"])
    plt.title(pair)

    file = f"{pair}.png"
    plt.savefig(file)
    plt.close()
    return file

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
                    await ch.send(fmt(d))

        await asyncio.sleep(900)

# =========================
# SLASH COMMANDS
# =========================
@client.tree.command(name="signal")
async def signal(interaction: discord.Interaction):
    d = generate("BTCUSD")
    await interaction.response.send_message(fmt(d))


@client.tree.command(name="show_trend_chart")
async def show(interaction: discord.Interaction, pair: str):
    df = get_data(PAIRS[pair])
    file = chart(df, pair)
    await interaction.response.send_message(file=discord.File(file))


@client.tree.command(name="auto_toggle")
async def toggle(interaction: discord.Interaction):
    global AUTO_SIGNAL
    AUTO_SIGNAL = not AUTO_SIGNAL
    await interaction.response.send_message(f"AUTO SIGNAL: {AUTO_SIGNAL}")

# =========================
# START
# =========================
@client.event
async def on_ready():
    await client.tree.sync()
    print("INSTITUTIONAL AI BOT READY")

asyncio.get_event_loop().create_task(loop())

client.run(TOKEN)
