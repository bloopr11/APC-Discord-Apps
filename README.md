📌 APC Apps — Discord (by Yor)

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
