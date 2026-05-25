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
    try:
        with open("adaylar.txt", "r", encoding="utf-8") as f:
            hisseler = list(dict.fromkeys([
                x.strip().upper()
                for x in f.readlines()
                if x.strip()
            ]))
    except Exception:
        hisseler = []

    if not hisseler:
        send_message("❌ adaylar.txt boş veya bulunamadı")
        return

    client = TelegramClient(
        StringSession(TG_SESSION),
        TG_API_ID,
        TG_API_HASH
    )

    client.start()

    for hisse in hisseler[:10]:


        client.send_message(BOT_USERNAME, f"/takas {hisse}")
        time.sleep(10)


        onceki_id = client.get_messages(BOT_USERNAME, limit=1)[0].id

        mesajlar = client.get_messages(BOT_USERNAME, limit=5)

        for msg in mesajlar:
    if msg.buttons and msg.text and hisse in msg.text:
        try:
            msg.click(text="7G")
            time.sleep(8)

            yeni_mesajlar = client.get_messages(BOT_USERNAME, limit=5)

            for yeni in yeni_mesajlar:
                if yeni.id > onceki_id and yeni.media:
                    dosya = client.download_media(yeni.media)
                    if dosya:
                        send_photo(dosya, f"📊 {hisse} 7G Takas Görseli")
                    break

            break

        except Exception as e:
            print("7G tıklama hatası:", e)
        

        takas_mesajlari = client.get_messages(BOT_USERNAME, limit=8)
        takas = "Takas cevabı bulunamadı"

        for msg in takas_mesajlari:
            if msg.text and "Takas" in msg.text:
                takas = msg.text
                break

        guclu_takas = (
            "BofA" in takas or
            "İş" in takas or
            "Garanti" in takas or
            "Yapı Kr." in takas or
            "Ziraat" in takas
        )



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
