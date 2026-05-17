import discord
import asyncio
from config import DISCORD_TOKEN, CHANNEL_ID, PAIR_LIST, TIMEFRAME
from indicators import get_ai_signal

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


def format_signal(data):
    return f"""
📊 {data['pair']} SIGNAL (AI ANALYSIS)

📈 Trend: {data['trend']}
🧠 Confidence AI: {data['confidence']}%

🟢 Action: {data['action']}
Entry: {data['entry']}
TP: {data['tp']}
SL: {data['sl']}

⏱ Timeframe: {TIMEFRAME}
"""


async def send_signal(channel):
    for pair in PAIR_LIST:
        data = get_ai_signal(pair)
        msg = format_signal(data)
        await channel.send(msg)


@client.event
async def on_ready():
    print(f"Bot aktif sebagai {client.user}")

    channel = client.get_channel(CHANNEL_ID)

    while True:
        await send_signal(channel)
        await asyncio.sleep(900)  # 15 menit sekali


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if message.content == "!signal":
        data = get_ai_signal("XAUUSD")
        await message.channel.send(format_signal(data))


client.run(DISCORD_TOKEN)
