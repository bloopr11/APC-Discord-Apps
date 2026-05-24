import asyncio
import threading
import time
import json
import random
import string
from typing import Optional, Dict, Any
import pandas as pd

class TVConnectionPool:
    def __init__(self, max_connections: int = 3, ttl: int = 300):
        self.max_connections = max_connections
        self.ttl = ttl
        self._pool: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def _make_key(self, exchange: str, symbol: str, interval: str) -> str:
        return f"{exchange}:{symbol}:{interval}"
    
    def _tv_token(self) -> str:
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    
    def _tv_msg(self, func: str, args: list) -> str:
        payload = json.dumps({"m": func, "p": args}, separators=(",", ":"))
        return f"~m~{len(payload)}~m~{payload}"
    
    def _tv_parse(self, raw: str) -> list:
        import re
        msgs = []
        for chunk in re.findall(r"~m~\d+~m~(.+?)(?=~m~\d+~m~|$)", raw, re.DOTALL):
            chunk = chunk.strip()
            if not chunk or chunk.startswith("~h~"):
                continue
            try:
                msgs.append(json.loads(chunk))
            except Exception:
                pass
        return msgs
    
    async def _create_connection(self, exchange: str, symbol: str, interval: str):
        import websockets
        headers = {
        "Origin": "https://www.tradingview.com",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    try:
        ws = await websockets.connect(
            "wss://data.tradingview.com/socket.io/websocket",
            extra_headers=headers,
            open_timeout=10
        )
        return ws
    except Exception as e:
        if "403" in str(e) and retry < 2:
            await asyncio.sleep(5 * (retry + 1))
            return await self._create_connection(exchange, symbol, interval, retry+1)
        raise
    
    async def get_connection(self, exchange: str, symbol: str, interval: str):
        key = self._make_key(exchange, symbol, interval)
        with self._lock:
            if key in self._pool:
                info = self._pool[key]
                if info.get('ws') and time.time() - info.get('last_used', 0) < self.ttl:
                    info['last_used'] = time.time()
                    return info['ws']
                else:
                    await self._close_connection(key)
            ws = await self._create_connection(exchange, symbol, interval)
            self._pool[key] = {'ws': ws, 'last_used': time.time()}
            return ws
    
    async def _close_connection(self, key: str):
        if key in self._pool:
            try:
                await self._pool[key]['ws'].close()
            except:
                pass
            del self._pool[key]
    
    async def close_all(self):
        for key in list(self._pool.keys()):
            await self._close_connection(key)
    
    def get_stats(self):
        with self._lock:
            return {'active': len(self._pool), 'max': self.max_connections}

_pool_instance = None

def init_tv_pool():
    global _pool_instance
    _pool_instance = TVConnectionPool(max_connections=3, ttl=300)
    print("✅ TV Connection pool initialized")

async def close_tv_pool():
    if _pool_instance:
        await _pool_instance.close_all()
        print("✅ TV Connection pool closed")

def fetch_tv_ws_sync(exchange: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    """Sync wrapper untuk async fetch dengan connection pool."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_fetch_tv_ws_async(exchange, symbol, interval, bars))
    finally:
        loop.close()

async def _fetch_tv_ws_async(exchange: str, symbol: str, interval: str, bars: int) -> pd.DataFrame:
    global _pool_instance
    if _pool_instance is None:
        init_tv_pool()
    ws = await _pool_instance.get_connection(exchange, symbol, interval)
    cs_token = f"cs_{_pool_instance._tv_token()}"
    await ws.send(_pool_instance._tv_msg("create_series", [cs_token, "sds_1", "s1", "sds_sym_1", interval, bars, ""]))
    candles = []
    deadline = time.time() + 20
    completed = False
    while time.time() < deadline and not completed:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=6)
        except asyncio.TimeoutError:
            break
        if "~h~" in raw:
            await ws.send(f"~m~{len(raw)}~m~{raw}")
            continue
        for msg in _pool_instance._tv_parse(raw):
            m = msg.get("m", "")
            if m == "timescale_update":
                bars_data = ((msg.get("p") or [{}])[1] or {}).get("sds_1", {}).get("s", [])
                for b in bars_data:
                    v = b.get("v", [])
                    if len(v) >= 5:
                        candles.append({
                            "ts": v[0], "Open": v[1], "High": v[2],
                            "Low": v[3], "Close": v[4], "Volume": v[5] if len(v) > 5 else 0,
                        })
            elif m == "series_completed":
                completed = True
                break
            elif m == "series_error":
                raise ValueError(f"TV error: {msg}")
    if not candles:
        raise ValueError("No candles")
    df = pd.DataFrame(candles)
    df["Date"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.set_index("Date").drop(columns=["ts"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
