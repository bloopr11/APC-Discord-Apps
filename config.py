import os

# Discord
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID", "0"))

# Trading config
PAIR_LIST = os.getenv("PAIR_LIST", "XAUUSD,BTCUSD").split(",")

TIMEFRAME = os.getenv("TIMEFRAME", "15M")
