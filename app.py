from flask import Flask, request
import requests

app = Flask(__name__)

WEBHOOK_URL = "WEBHOOK_DISCORD_KAMU"

@app.route('/')

def home():

    return "BOT ONLINE"

@app.route('/webhook', methods=['POST'])

def webhook():

    data = request.json

    signal = data.get("signal")

    symbol = data.get("symbol")

    price = data.get("price")

    trend = data.get("trend")

    emoji = "🚀"

    color = 65280

    if signal == "SELL":

        emoji = "🔻"

        color = 16711680

    embed = {

        "title": f"{emoji} {symbol} LUX SIGNAL",

        "color": color,

        "fields": [

            {
                "name": "Signal",
                "value": signal,
                "inline": True
            },

            {
                "name": "Trend",
                "value": trend,
                "inline": True
            },

            {
                "name": "Price",
                "value": str(price),
                "inline": False
            }

        ]

    }

    requests.post(

        WEBHOOK_URL,

        json={
            "embeds": [embed]
        }

    )

    return {
        "status": "ok"
    }

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=10000

    )
