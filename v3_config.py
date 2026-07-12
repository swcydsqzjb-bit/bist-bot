import os
from zoneinfo import ZoneInfo


# ============================================================
# GENEL AYARLAR
# ============================================================

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

BOT_MODE = os.getenv("BOT_MODE", "auto").strip().lower()

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


# ============================================================
# DOSYALAR
# ============================================================

SIGNALS_FILE = "v3_signals_history.csv"
DAILY_RESULTS_FILE = "v3_daily_results.csv"
TODAY_CANDIDATES_FILE = "v3_today_candidates.csv"


# ============================================================
# VERİ AYARLARI
# ============================================================

SYMBOL_SOURCE_URL = (
    "https://stockanalysis.com/list/borsa-istanbul/"
)

DAILY_PERIOD = "3y"
DAILY_INTERVAL = "1d"

MIN_DAILY_BARS = 120

# Çok düşük işlem hacimli hisseleri elemek için yaklaşık eşik.
# Son 20 günlük ortalama TL hacmi bu değerin altında olanlar elenir.
MIN_AVG_TURNOVER_TL = 5_000_000


# ============================================================
# SMART MONEY FİLTRELERİ
# ============================================================

# Hisse çok yükselmişse yeni aday olarak gösterilmesin.
MAX_RETURN_5D = 15.0
MAX_RETURN_20D = 30.0
MAX_EMA20_DISTANCE = 15.0
MAX_RSI = 75.0

# Hacim birikimi
MIN_VOLUME_RATIO = 1.10
STRONG_VOLUME_RATIO = 1.50

# Sağlıklı RSI bölgesi
MIN_HEALTHY_RSI = 45.0
MAX_HEALTHY_RSI = 68.0

# Sıkışma sınırları
MAX_RANGE_20D = 22.0
STRONG_COMPRESSION_RANGE = 15.0

# Hissenin günün zirvesine yakın kapanması
MIN_CLOSE_POSITION = 0.60

# Uzun üst fitil sınırı
MAX_UPPER_WICK_RATIO = 0.40


# ============================================================
# PUANLAMA
# ============================================================

MIN_SMART_MONEY_SCORE = 65

# Telegram’a en fazla kaç aday gönderilsin?
MAX_DAILY_CANDIDATES = 3

# Çok güçlü tek aday eşiği
ELITE_SCORE = 82


# ============================================================
# PERFORMANS TAKİBİ
# ============================================================

CHECK_DAYS = [1, 3, 5, 10]

# Başarı değerlendirmesi
SUCCESS_RETURN_3D = 3.0
SUCCESS_RETURN_5D = 5.0

# Aynı hisse aynı gün bir kez kaydedilir.
ALLOW_DUPLICATE_DAILY_SIGNAL = False


# ============================================================
# ÇALIŞMA SAATLERİ
# ============================================================

DAILY_SCAN_HOUR = 9
DAILY_SCAN_MINUTE_START = 25
DAILY_SCAN_MINUTE_END = 45

INTRADAY_START_HOUR = 10
INTRADAY_END_HOUR = 18
