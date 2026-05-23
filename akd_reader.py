import os
import time
import requests
from telethon.sync import TelegramClient
from telethon.sessions import StringSession

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TG_API_ID = int(os.getenv("TG_API_ID"))
TG_API_HASH = os.getenv("TG_API_HASH")
TG_SESSION = os.getenv("TG_SESSION")

def send_message(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text}
    )
    print("Telegram cevap:", r.status_code, r.text, flush=True)

client = TelegramClient(
    StringSession(TG_SESSION),
    TG_API_ID,
    TG_API_HASH
)

with client:
    send_message("🧪 AKD Reader test başladı")

    hisse = "SNICA"

    client.send_message(
        "ucretsizderinlikbot",
        f"/takas {hisse}"
    )
    time.sleep(5)

    client.send_message(
        "ucretsizderinlikbot",
        f"/akd {hisse}"
    )

    mesajlar = client.get_messages("ucretsizderinlikbot", limit=1)

    if mesajlar:
        cevap = mesajlar[0].text
        send_message("🧪 TAKAS CEVABI:\n" + str(cevap)[:3500])
    else:
        send_message("❌ Takas cevabı bulunamadı")
