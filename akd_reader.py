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

BOT_USERNAME = "ucretsizderinlikbot"


def send_message(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text}
    )
    print("Telegram mesaj:", r.status_code, r.text, flush=True)


def send_photo(photo_path, caption=""):
    with open(photo_path, "rb") as photo:
        r = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption},
            files={"photo": photo}
        )
    print("Telegram foto:", r.status_code, r.text, flush=True)


def main():
    hisse = "SNICA"

    client = TelegramClient(
        StringSession(TG_SESSION),
        TG_API_ID,
        TG_API_HASH
    )

    client.start()

    send_message("🧪 AKD/Takas test başladı")

    client.send_message(BOT_USERNAME, f"/takas {hisse}")
    time.sleep(10)

    takas_mesajlari = client.get_messages(BOT_USERNAME, limit=8)
    takas = "Takas cevabı bulunamadı"

    for msg in takas_mesajlari:
        if msg.text and "Takas" in msg.text:
            takas = msg.text
            break

    send_message("📊 TAKAS CEVABI:\n" + str(takas)[:3500])

    client.send_message(BOT_USERNAME, f"/akd {hisse}")
    time.sleep(20)

    akd_mesajlari = client.get_messages(BOT_USERNAME, limit=10)
    akd_bulundu = False

    for msg in akd_mesajlari:
        if msg.media:
            dosya = client.download_media(msg, file="akd_gorsel.png")
            if dosya:
                send_photo(dosya, f"🏦 {hisse} AKD Görseli")
                akd_bulundu = True
                break

    if not akd_bulundu:
        send_message("❌ AKD görseli bulunamadı")

    client.disconnect()


if __name__ == "__main__":
    main()
