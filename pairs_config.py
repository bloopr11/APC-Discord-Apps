# ==================================================
# pairs_config.py  —  Konfigurasi Pairs & Timeframes
#
# Edit file ini untuk menambah/hapus/ubah pair.
# app.py otomatis membaca dari sini.
# ==================================================

PAIRS = {
    # ── Crypto ────────────────────────────────────
    "BTCUSD":  {"tv": ("BINANCE",  "BTCUSDT"),  "yf": "BTC-USD"},
    "ETHUSD":  {"tv": ("BINANCE",  "ETHUSDT"),  "yf": "ETH-USD"},
    "XRPUSD":  {"tv": ("BINANCE",  "XRPUSDT"),  "yf": "XRP-USD"},
    "SOLUSD":  {"tv": ("BINANCE",  "SOLUSDT"),  "yf": "SOL-USD"},
    "ADAUSD":  {"tv": ("BINANCE",  "ADAUSDT"),  "yf": "ADA-USD"},
    "DOGEUSD": {"tv": ("BINANCE",  "DOGEUSDT"), "yf": "DOGE-USD"},
    "BNBUSD":  {"tv": ("BINANCE",  "BNBUSDT"),  "yf": "BNB-USD"},
    "AVAXUSD": {"tv": ("BINANCE",  "AVAXUSDT"), "yf": "AVAX-USD"},
    "LINKUSD": {"tv": ("BINANCE",  "LINKUSDT"), "yf": "LINK-USD"},
    "DOTUSD":  {"tv": ("BINANCE",  "DOTUSDT"),  "yf": "DOT-USD"},

    # ── Komoditas ─────────────────────────────────
    "XAUUSD":  {"tv": ("OANDA",    "XAUUSD"),   "yf": "GC=F"},
    "XAGUSD":  {"tv": ("OANDA",    "XAGUSD"),   "yf": "SI=F"},

    # ── Forex ─────────────────────────────────────
    "EURUSD":  {"tv": ("OANDA",    "EURUSD"),   "yf": "EURUSD=X"},
    "GBPUSD":  {"tv": ("OANDA",    "GBPUSD"),   "yf": "GBPUSD=X"},
    "USDJPY":  {"tv": ("OANDA",    "USDJPY"),   "yf": "USDJPY=X"},

    # ── Tambah pair baru di sini ──────────────────
    # "TONUSD":  {"tv": ("BINANCE",  "TONUSDT"),  "yf": "TON11419-USD"},
}

TIMEFRAMES = {
    "5m":  {"tv_interval": "5",   "yf_period": "5d",   "yf_interval": "5m",  "bars": 500},
    "15m": {"tv_interval": "15",  "yf_period": "7d",   "yf_interval": "15m", "bars": 500},
    "1h":  {"tv_interval": "60",  "yf_period": "30d",  "yf_interval": "1h",  "bars": 500},
    "4h":  {"tv_interval": "240", "yf_period": "60d",  "yf_interval": "4h",  "bars": 300},
}

MTF_FETCH = {
    "1h":  {"tv_interval": "60",  "yf_period": "30d",  "yf_interval": "1h",  "bars": 300},
    "4h":  {"tv_interval": "240", "yf_period": "60d",  "yf_interval": "4h",  "bars": 200},
    "1d":  {"tv_interval": "1D",  "yf_period": "180d", "yf_interval": "1d",  "bars": 200},
    "1wk": {"tv_interval": "1W",  "yf_period": "730d", "yf_interval": "1wk", "bars": 100},
}

# Pair yang tetap aktif saat sesi ASIA
ASIA_ACTIVE = {
    "BTCUSD", "ETHUSD", "BNBUSD", "ADAUSD", "XRPUSD",
    "SOLUSD", "DOGEUSD", "AVAXUSD", "LINKUSD", "DOTUSD",
}

DEFAULT_TF = "5m"