import matplotlib
matplotlib.use("Agg")
import discord
from discord import app_commands
import asyncio
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ======================
# CONFIG (ENV SAFE)
# ======================
TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

PAIRS = ["XAUUSD","BTCUSD","ETHUSD","DOGEUSD","SOLUSD","ADAUSD","XRPUSD","KASUSD"]
TIMEFRAME = "15M"

AUTO_SIGNAL = False
last_signal_time = {}

# ======================
# DISCORD CLIENT
# ======================
class Bot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

client = Bot()

# ======================
# FAKE MARKET DATA (replace with API later)
# ======================
def generate_fake_data():
    data = np.cumsum(np.random.randn(100)) + 2000
    df = pd.DataFrame({
        "open": data + np.random.randn(100),
        "high": data + np.random.rand(100)*2,
        "low": data - np.random.rand(100)*2,
        "close": data
    })
    return df

# ======================
# INDICATORS
# ======================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    return macd_line, signal

# ======================
# SMC LOGIC
# ======================
def smc(df):
    last = df.iloc[-1]
    prev = df.iloc[-2]

    bos = last["close"] > prev["high"]
    choch = last["close"] < prev["low"]

    liquidity_sweep = last["high"] > df["high"].rolling(10).max().iloc[-2]

    return {
        "BOS": bool(bos),
        "CHoCH": bool(choch),
        "LS": bool(liquidity_sweep)
    }

# ======================
# WINRATE AI MODEL
# ======================
def winrate(rsi_val, macd_val, smc_val):
    score = 50

    if rsi_val < 30:
        score += 15
    if rsi_val > 70:
        score -= 15

    if macd_val > 0:
        score += 15
    else:
        score -= 10

    if smc_val["BOS"]:
        score += 10
    if smc_val["LS"]:
        score += 10
    if smc_val["CHoCH"]:
        score -= 10

    return max(0, min(100, score))

# ======================
# SIGNAL ENGINE
# ======================
def generate_signal(pair):
    df = generate_fake_data()
    close = df["close"]

    r = rsi(close).iloc[-1]
    m_line, m_sig = macd(close)
    m = m_line.iloc[-1] - m_sig.iloc[-1]

    s = smc(df)
    w = winrate(r, m, s)

    action = "BUY" if w >= 55 else "SELL"

    return {
        "pair": pair,
        "rsi": round(r,2),
        "macd": round(m,2),
        "smc": s,
        "winrate": w,
        "action": action,
        "entry": round(close.iloc[-1],2),
        "tp": round(close.iloc[-1] + random.uniform(10,30),2),
        "sl": round(close.iloc[-1] - random.uniform(10,30),2),
    }

# ======================
# FORMAT MESSAGE
# ======================
def format_signal(data):
    return f"""
📊 {data['pair']} PRO SIGNAL (AI MODEL)

🧠 Winrate AI: {data['winrate']}%

📈 RSI: {data['rsi']}
📊 MACD: {data['macd']}

🏗 BOS: {data['smc']['BOS']}
🔄 CHoCH: {data['smc']['CHoCH']}
💧 Liquidity Sweep: {data['smc']['LS']}

🟢 Action: {data['action']}
Entry: {data['entry']}
TP: {data['tp']}
SL: {data['sl']}

⏱ TF: {TIMEFRAME}
"""

# ======================
# CHART GENERATOR
# ======================
def make_chart(df, pair):
    plt.figure(figsize=(8,4))
    plt.plot(df["close"], label="Close")
    plt.title(pair)
    plt.legend()

    file = f"{pair}_chart.png"
    plt.savefig(file)
    plt.close()
    return file

# ======================
# AUTO SIGNAL FILTER
# ======================
def can_send(pair, cooldown=900):
    now = asyncio.get_event_loop().time()

    if pair not in last_signal_time:
        last_signal_time[pair] = now
        return True

    if now - last_signal_time[pair] > cooldown:
        last_signal_time[pair] = now
        return True

    return False

# ======================
# AUTO LOOP
# ======================
async def auto_loop():
    await client.wait_until_ready()
    channel = client.get_channel(CHANNEL_ID)

    while True:
        if AUTO_SIGNAL and channel:
            for p in PAIRS:
                sig = generate_signal(p)

                if sig["winrate"] >= 75 and can_send(p):
                    await channel.send(format_signal(sig))

        await asyncio.sleep(900)

# ======================
# SLASH COMMANDS
# ======================

@client.tree.command(name="signal")
async def signal(interaction: discord.Interaction):
    data = generate_signal("XAUUSD")
    await interaction.response.send_message(format_signal(data))


@client.tree.command(name="show_trend_chart")
@app_commands.describe(pair="Pair symbol")
async def chart(interaction: discord.Interaction, pair: str):
    pair = pair.upper()

    df = generate_fake_data()
    file = make_chart(df, pair)

    await interaction.response.send_message(file=discord.File(file))


@client.tree.command(name="timeframe_set")
async def tf(interaction: discord.Interaction, tf: str):
    global TIMEFRAME
    TIMEFRAME = tf.upper()
    await interaction.response.send_message(f"⏱ Timeframe set ke {TIMEFRAME}")


@client.tree.command(name="auto_toggle")
async def toggle(interaction: discord.Interaction):
    global AUTO_SIGNAL
    AUTO_SIGNAL = not AUTO_SIGNAL

    status = "ON 🔥" if AUTO_SIGNAL else "OFF ❌"
    await interaction.response.send_message(f"Auto Signal: {status}")

# ======================
# READY EVENT
# ======================
@client.event
async def on_ready():
    await client.tree.sync()
    print(f"BOT READY: {client.user}")

# START AUTO LOOP
asyncio.get_event_loop().create_task(auto_loop())

client.run(TOKEN)
