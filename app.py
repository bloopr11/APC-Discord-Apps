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

# ======================
# ENV
# ======================
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

PAIRS = {
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XAUUSD": "GC=F",   # Gold Futures (TradingView style equivalent)
    "XRPUSD": "XRP-USD",
    "SOLUSD": "SOL-USD"
}

TIMEFRAME = "15m"
AUTO_SIGNAL = False

# ======================
# DISCORD CLIENT
# ======================
class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

client = Bot()

# ======================
# TRADINGVIEW STYLE DATA (YFINANCE)
# ======================
def get_data(symbol):
    df = yf.download(symbol, period="5d", interval="15m")
    df = df.dropna()
    return df

# ======================
# INDICATORS
# ======================
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

# ======================
# SMC (REALISTIC SIMPLIFIED)
# ======================
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

# ======================
# WINRATE MODEL
# ======================
def winrate(rsi_v, macd_v, smc_v):
    score = 50

    if rsi_v < 30:
        score += 15
    if rsi_v > 70:
        score -= 15

    if macd_v > 0:
        score += 15
    else:
        score -= 10

    if smc_v["BOS"]:
        score += 10
    if smc_v["LIQ"]:
        score += 10
    if smc_v["CHoCH"]:
        score -= 10

    return max(0, min(100, score))

# ======================
# SIGNAL ENGINE
# ======================
def generate_signal(pair):
    symbol = PAIRS[pair]
    df = get_data(symbol)

    close = df["Close"]

    r = rsi(close).iloc[-1]
    macd_line, macd_sig = macd(close)
    m = macd_line.iloc[-1] - macd_sig.iloc[-1]

    s = smc(df)
    w = winrate(r, m, s)

    action = "BUY" if w >= 55 else "SELL"

    price = float(close.iloc[-1])

    return {
        "pair": pair,
        "rsi": round(r,2),
        "macd": round(m,2),
        "smc": s,
        "winrate": w,
        "action": action,
        "entry": price,
        "tp": price + 50,
        "sl": price - 50
    }

# ======================
# FORMAT MESSAGE
# ======================
def fmt(d):
    return f"""
📊 {d['pair']} TRADINGVIEW AI SIGNAL

🧠 Winrate: {d['winrate']}%

📈 RSI: {d['rsi']}
📊 MACD: {d['macd']}

🏗 BOS: {d['smc']['BOS']}
🔄 CHoCH: {d['smc']['CHoCH']}
💧 LIQ: {d['smc']['LIQ']}

🟢 Action: {d['action']}
Entry: {d['entry']}
TP: {d['tp']}
SL: {d['sl']}

⏱ TF: {TIMEFRAME}
"""

# ======================
# CHART (CANDLESTYLE SIMPLE)
# ======================
def chart(df, pair):
    plt.figure(figsize=(10,4))
    plt.plot(df["Close"])
    plt.title(pair)
    file = f"{pair}.png"
    plt.savefig(file)
    plt.close()
    return file

# ======================
# SLASH COMMANDS
# ======================
@client.tree.command(name="signal")
async def signal(interaction: discord.Interaction):
    d = generate_signal("BTCUSD")
    await interaction.response.send_message(fmt(d))


@client.tree.command(name="show_trend_chart")
async def show(interaction: discord.Interaction, pair: str):
    df = get_data(PAIRS[pair])
    file = chart(df, pair)
    await interaction.response.send_message(file=discord.File(file))


@client.tree.command(name="timeframe_set")
async def tf(interaction: discord.Interaction, tf: str):
    global TIMEFRAME
    TIMEFRAME = tf
    await interaction.response.send_message(f"TF set: {TIMEFRAME}")


@client.tree.command(name="auto_toggle")
async def toggle(interaction: discord.Interaction):
    global AUTO_SIGNAL
    AUTO_SIGNAL = not AUTO_SIGNAL
    await interaction.response.send_message(f"AUTO: {AUTO_SIGNAL}")

# ======================
# AUTO LOOP
# ======================
async def loop():
    await client.wait_until_ready()
    ch = client.get_channel(CHANNEL_ID)

    while True:
        if AUTO_SIGNAL:
            for p in PAIRS:
                d = generate_signal(p)
                if d["winrate"] >= 75:
                    await ch.send(fmt(d))

        await asyncio.sleep(900)

@client.event
async def on_ready():
    await client.tree.sync()
    print("BOT READY (TRADINGVIEW MODE)")

asyncio.get_event_loop().create_task(loop())

client.run(TOKEN)
