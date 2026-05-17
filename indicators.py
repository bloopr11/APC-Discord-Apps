import random

def get_ai_signal(pair):
    # simulasi AI scoring (bisa nanti diganti ML beneran)
    confidence = random.randint(55, 92)

    if confidence >= 70:
        action = "BUY"
        trend = "Bullish"
    else:
        action = "SELL"
        trend = "Bearish"

    entry = round(random.uniform(2000, 2100), 2)
    tp = round(entry + random.uniform(5, 20), 2)
    sl = round(entry - random.uniform(5, 20), 2)

    return {
        "pair": pair,
        "trend": trend,
        "confidence": confidence,
        "action": action,
        "entry": entry,
        "tp": tp,
        "sl": sl
    }
