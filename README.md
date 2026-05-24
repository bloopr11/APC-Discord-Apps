# 🚀 APC-Discord v4

Advanced multi-factor trading signal engine for Discord using:
- Smart Money Concept (SMC)
- Multi Timeframe Bias
- Orderflow Proxy
- Fair Value Gap (FVG)
- Adaptive AI Scoring
- Regime Detection
- Liquidity Sweep Protection
- ML-based score optimization

Supports:
- Crypto
- Gold (XAUUSD)
- Futures pairs
- Discord auto signals
- TradingView WebSocket
- Yahoo Finance fallback

---

# ✨ Features

## 📊 Institutional Signal Engine
- EMA trend stack
- RSI divergence
- MACD momentum
- Stochastic filter
- Bollinger positioning
- ATR volatility

---

## 🧠 AI Adaptive Scoring
Dynamic score engine with:
- Rule-based scoring
- ML adaptive weighting
- Market regime adjustment
- Confidence calculation
- Confluence analysis

---

## 🏦 Smart Money Concept (SMC)
Includes:
- BOS detection
- CHoCH
- Order Block proxy
- Liquidity zones
- Liquidity pressure

---

## 📐 Fair Value Gap (FVG)
- Bullish imbalance detection
- Bearish imbalance detection
- Nearest FVG mapping
- FVG confluence scoring

---

## 🌍 Multi Timeframe Bias
Automatic higher timeframe alignment:
- 5m → 1h + 4h
- 15m → 1h + 4h
- 1h → 4h + 1D
- 4h → 1D + 1W

---

## ⏰ Session Filter
Session-aware trading:
- Asia
- London
- New York
- Overlap

Special logic:
- XAUUSD session optimization
- Crypto Asia-session support

---

## 📦 Orderflow Proxy
Estimated institutional pressure using:
- Delta volume
- CVD slope
- Buy/sell volume ratio
- Absorption detection

---

## 🛡 Anti Liquidity Sweep SL
Advanced TP/SL system:
- ATR-based stop
- Liquidity pool avoidance
- Anti-hunt stop placement
- Multi-target RR

---

# 🤖 ML Retraining
Built-in:
- Signal logging
- Score history
- Outcome tracking
- Adaptive retraining

---

# 📷 Discord Signal Example

Signals include:
- AI confidence score
- TP1 / TP2
- SL1 / SL2
- RR ratio
- Session info
- MTF bias
- Orderflow analysis
- FVG zones
- Market regime
- Chart image

---

# 🧩 Supported Assets

## Crypto
- BTCUSD
- ETHUSD
- XRPUSD
- SOLUSD
- ADAUSD
- DOGEUSD
- AVAXUSD
- LINKUSD
- BNBUSD

## Metals
- XAUUSD
- XAGUSD

## Futures Perpetual
- BTCUSDT.P
- ETHUSDT.P
- XRPUSDT.P
- SOLUSDT.P
- ADAUSDT.P
- BNBUSDT.P
- DOGEUSDT.P

---

# 📊 Signal Scoring System

| Score | Confidence |
|---|---|
| 82+ | VERY HIGH |
| 70+ | HIGH |
| 55+ | MODERATE |
| <55 | LOW |

---

# ⚠️ Disclaimer

This project is an educational and analytical trading tool.

Trading involves risk.  
No guarantees of profitability are provided.

Always use proper risk management.

---


# 👨‍💻 Author
Bloopr11


---


📌 APC Apps — Discord (by Yor) v1

Bot ini adalah AI-powered indikator assistant yang berjalan di Discord. Ia menggabungkan analisis teknikal, orderflow proxy, multi-timeframe confluence, serta rule-based scoring untuk menghasilkan sinyal trading otomatis dengan transparansi penuh.


⚙️ Fitur Utama
1. Market Data Fetching
Menggunakan WebSocket untuk mengambil data harga real-time.
Mendukung berbagai pasangan: BTCUSD, ETHUSD, XAUUSD, XRPUSD, SOLUSD, dll.
Timeframe fleksibel: 5m, 15m, 1h, 4h.
2. Indicators Engine
EMA, RSI, MACD, ATR, Stochastic, Bollinger Bands → indikator klasik untuk tren & momentum.
Volume Ratio → membandingkan volume saat ini dengan rata-rata historis.
3. Delta Orderflow Proxy
Menghitung Cumulative Volume Delta (CVD) untuk mendeteksi dominasi buyer/seller.
Menentukan bias: BUY / SELL / NEUTRAL.
Deteksi absorption (smart money masuk saat harga stagnan).
4. Fair Value Gap (FVG) Engine
Mendeteksi imbalance zone (gap harga).
Memberi tahu apakah harga sedang berada di zona FVG bullish/bearish.
5. Multi-Timeframe Bias (MTF)
Analisis konfluensi dari timeframe lebih tinggi (misalnya 1h, 4h, 1d).
Menggunakan kombinasi EMA stack, RSI, MACD, dan posisi harga terhadap EMA200.
6. Session Filter
Menentukan sesi aktif (Asia, London, New York, Overlap).
Memberikan score modifier sesuai likuiditas.
Menyesuaikan penalti/bonus untuk pair tertentu (misalnya XAUUSD hanya aktif London/NY).
7. Smart Money Concepts (SMC)
Deteksi Break of Structure (BOS), Change of Character (CHoCH), Liquidity Grab, dan Order Block.
8. Liquidity & Flow Engine
Menentukan zona likuiditas (HIGH/LOW zone).
Mengukur arah tekanan pasar (BUY/SELL).
Flow momentum sederhana untuk bias arah.
9. Trend Strength (ADX Proxy)
Mengukur kekuatan tren dengan directional movement (+DI / -DI).
Memberi skor tren 0–100.
10. Advanced Rule-Based Scoring
Sistem scoring transparan (0–100) dengan kategori:
Trend, Momentum, SMC, Orderflow, FVG, MTF, Session, Bollinger Bands
Memberikan label confidence: 🔥 Very High, ✅ High, ⚠️ Moderate, ❌ Low.
Menentukan action BUY/SELL dengan TP/SL berbasis anti-liquidity sweep.
11. Signal Generation & Discord Embed
Pipeline lengkap untuk menghasilkan sinyal trading:
Entry, TP, SL, Risk-Reward.
Confidence & confluence breakdown.
Format Discord Embed dengan warna & emoji (🟢 BUY / 🔴 SELL).


🚀 Alur Kerja Bot

Ambil data harga dari WebSocket.
Hitung indikator dasar (RSI, MACD, ATR, dll).
Jalankan engine tambahan (Orderflow, FVG, MTF, Session).
Skoring sinyal dengan advanced_score().
Tentukan aksi BUY/SELL + TP/SL.
Kirim sinyal ke Discord channel dalam bentuk Embed.


🎯 Tujuan

Bot ini dirancang untuk: Memberikan sinyal AI prediction otomatis dengan transparansi penuh.
Menggabungkan indikator klasik + konsep smart money.
Membantu trader dalam pengambilan keputusan cepat di pasar crypto & emas.



_______________________________________________________________________________________________



🔑 Changelog

1. Scoring Engine
Before; Menggunakan fungsi ai_score sederhana berbasis indikator klasik (RSI, MACD, EMA, SMC, dll).
After; Menggantikan dengan advanced_score yang jauh lebih kompleks Breakdown kategori: TREND, MOMENTUM, SMC, ORDERFLOW, FVG, MTF, SESSION, BB.
Baseline score 50 + penambahan/pengurangan per kategori.
Menghitung` confluence count (berapa kategori yang sejalan dengan arah BUY/SELL).
Transparansi skor per kategori.

2. Confidence Label
Before; Label sederhana: VERY HIGH, HIGH, MODERATE, LOW berdasarkan skor.
After; Label lebih detail dengan syarat skor + confluence:
🔥 VERY HIGH jika score ≥ 82 & confluence ≥ 5.
✅ HIGH jika score ≥ 70 & confluence ≥ 4.
⚠️ MODERATE jika score ≥ 55.
❌ LOW jika di bawahnya.

3. Risk Management (TP/SL)
Before; TP/SL berbasis ATR multiplier (TP = 2×ATR, SL = 1×ATR).
After; Menggunakan tp_sl_anti_sweep:
Anti-liquidity-sweep logic (menghindari SL di area pool likuiditas).
Menghitung RR berdasarkan jarak SL dinamis.
Memanfaatkan fungsi liquidity_pools.

4. New Engines & Filters
Menambahkan modul baru :
Orderflow Delta (delta_orderflow) → Approx real orderflow dengan cumulative volume delta.
Fair Value Gap (fvg_engine) → Deteksi imbalance/FVG zone.
Multi-Timeframe Bias (mtf_bias) → Konfluensi dari HTF (EMA, RSI, MACD, 200 EMA).
Session Filter (session_filter) → Menentukan sesi aktif (Asia, London, NY, Overlap) + score modifier.
Trend Strength (trend_strength) → Proxy ADX untuk kekuatan tren.
Liquidity Pools (liquidity_pools) → Identifikasi buy/sell side liquidity.
Flow Engine (flow) → Momentum rata-rata candle.
SMC Engine (smc) → BOS, CHoCH, Order Block, Liquidity.
EMA Confluence (ema_confluence) → Stack EMA 8/21/50/200.
RSI Divergence (rsi_divergence) → Deteksi bullish/bearish divergence.

5. Signal Generation
Before; generate_signal hanya menggabungkan indikator dasar + ai_score.
After; generate_signal:
Menggabungkan indikator dasar + semua engine baru.
Menggunakan advanced_score untuk hasil akhir.
Menambahkan informasi lengkap: confluence, confidence, sweep data, session, orderflow, FVG, MTF, dll.

6. Embed Output (Discord)
Before; Embed sederhana dengan indikator, AI analysis, TP/SL, dll.
After; Embed lebih kaya:
Menampilkan confluence count [x/6].
Menambahkan informasi session, orderflow, FVG, MTF, trend strength.
Confidence label lebih detail.