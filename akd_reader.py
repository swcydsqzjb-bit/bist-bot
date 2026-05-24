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

def send_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, "rb") as photo:
        r = requests.post(
            url,
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": photo}
        )
    print("Foto cevap:", r.status_code, r.text, flush=True)

client = TelegramClient(
    StringSession(TG_SESSION),
    TG_API_ID,
    TG_API_HASH
)

def son_cevabi_oku(aranan_kelime, limit=8):
    mesajlar = client.get_messages("ucretsizderinlikbot", limit=limit)

    for msg in mesajlar:
        if msg.text and aranan_kelime in msg.text:
            return msg.text

    return "Cevap bulunamadı"


with client:
    hisse = "SNICA"

    send_message("🧪 AKD/Takas test başladı")

    client.send_message("ucretsizderinlikbot", f"/takas {hisse}")
    time.sleep(10)
    takas = son_cevabi_oku("Takas")

    send_message("📊 TAKAS CEVABI:\n" + str(takas)[:3500])

    client.send_message("ucretsizderinlikbot", f"/akd {hisse}")
    time.sleep(15)
    akd = son_cevabi_oku("Aracı Kurum")

    client.send_message("ucretsizderinlikbot", f"/akd {hisse}")
time.sleep(15)

mesajlar = client.get_messages("ucretsizderinlikbot", limit=8)

akd_bulundu = False

for msg in mesajlar:
    if msg.photo:
        dosya = client.download_media(msg, file="akd.png")
        send_photo(dosya, f"🏦 {hisse} AKD Görseli")
        akd_bulundu = True
        break

if not akd_bulundu:
    send_message("❌ AKD görseli bulunamadı")
