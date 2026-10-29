# LARUS PEG Momentum

Bağımsız yeni filtre. Mevcut V8–V33, weekly_scan ve weekly_momentum stratejilerinin sonuçlarını değiştirmez.
Bu modülde RSI kullanılır; önceki RSI'siz motorlar aynı kalır.

## Mevcut durum

- CSV tarama motoru ve otomatik günlük fiyat indirme adaptörü hazır.
- Hisse evreni mevcut `weekly_scan.universe()` fonksiyonundan alınır. Otomatik liste eksikse tarama durur; eski V16 listesine sessizce düşmez.
- Yahoo'dan 5 yıllık günlük veriler alınır; OHLC'nin tamamı aynı düzeltme faktörüyle işlenir.
- İş Yatırım mali tablo servisinden otomatik bilanço indirme eklendi. Son açıklanan dört dönem sayfadaki dönem listesinden bulunur; takvimde bir çeyrek geçti diye bilanço açıklanmış sayılmaz.
- **İndirme, karşılaştırılabilir PEG verisi doğrulaması değildir.** Kaynak birikimli tablolar sunuyor ve bazı geçmiş yıl sonlarını güncel baza taşıyor. Yayın zamanı, tüm dönemlerin aynı satın alma gücü ve tek seferlik kâr kontrolleri henüz çözülemedi.
- Mevcut botun Yahoo EPS/FK verisi, karşılaştırılabilir esas faaliyet kârı ve nakit akışı yerine kullanılmaz.
- Bu nedenle doğrulanmış `peg_momentum/data/fundamentals.csv` yokken durum `INCOMPLETE_DATA` olur, aday listesi boş kalır ve süreç kod 2 ile çıkar. Rapor ve teknik ön izleme yine saklanır.
- Otomatik Telegram gönderimi, emir gönderimi, zamanlanmış çalışma ve geçmiş performans testi bu değişiklikte yoktur.

## Çalıştırma

Depo kökünde:

```bash
pip install -r requirements.txt
python -m unittest discover -s peg_momentum/tests -v
python -m peg_momentum.run
```

Yalnızca üç şirketle canlı kaynak kontrolü (fiyat/aday taraması yapmaz):

```bash
python -m peg_momentum.financial_source --symbols THYAO,ASELS,BIMAS
python -m peg_momentum.source_audit --source-csv peg_momentum_output/source_financials.csv
```

`--symbols` verilmezse mevcut botun hisse evreni kullanılır. Kaynak erişim hatalarında kod atlanmaz;
tanı dosyasına hata satırı yazılır. Otomatik indirme, doğrulanmış `data/fundamentals.csv` dosyasının üzerine yazmaz.
Üç şirket için yapılan canlı kontrol tüm BIST'in bilanço kapsamını kanıtlamaz.

30 Ekim 2026 kontrolü: THYAO 2026/06, ASELS 2026/09 ve BIMAS 2026/06 dönemlerindeki
satış, esas faaliyet kârı ve ana ortaklık net kârından oluşan 9 güncel kalem KAP özet tablosuyla
gösterim hassasiyeti içinde eşleşti. Aynı şirketlerin 2025/12 dönemlerindeki 9 kalemde fark görüldü.
İş Yatırım, son yıl sonu rakamlarının enflasyonla son finansal döneme taşındığını açıklıyor;
KAP özet sayfası ise cari dönem sütunlarını saklıyor ve sonraki düzeltmeleri içermediğini belirtiyor.
Bu nedenle farklar veri hatası olarak sınıflandırılmaz; karşılaştırma bazı incelemesi gerektirir.
KAP özetinden eski ve yeni kalemleri doğrudan birleştirerek TTM türetilmez.
Kontrol sonuçları: `validation/kap_sample_audit_2026-10-30.csv`.

`python -m peg_momentum.run --asof 2026-10-30` ile kapanmış seans tarihi seçilebilir.
Yerel İstanbul saati 19:00 öncesi bugünkü veriler kullanılmaz; hafta sonu cuma kullanılır.
Resmî tatil/yarım gün takvimi henüz entegre değildir. Haftalık mum cuma etiketiyle değerlendirilir;
henüz tamamlanmamış hafta kullanılmaz. Haftanın son seansı perşembeyse cuma tarihine kadar beklenir.
Bu bir anlık fiyat/takas taraması değildir.

GitHub Actions içinde **LARUS PEG Momentum** ayrı iş akışıdır. Pull request'te yalnızca internetsiz testler çalışır.
Elle tetiklemede fiyat taraması çalışır. Eksik bilanço yüzünden başarısız görünen taramanın tanı dosyaları da artifact olarak saklanır.
Mevcut `LARUS ALPHA FULL MARKET` iş akışının zamanlaması/Telegram zinciri değiştirilmez.

## Filtre

1. Tarihsel PEG: `F/K / EPS yıllık büyümesi (%)`; `0 < PEG < 1`. %30 büyüme için bölen 30'dur.
2. İki karşılaştırılabilir 12 aylık dönemde EPS/net kâr pozitif. EPS büyümesi %100 üzerindeyse düşük baz kontrolü için `REVIEW_REQUIRED`.
3. Son tek çeyrek esas faaliyet kârı, geçen yılın aynı tek çeyreğine göre en az %20 yüksek; iki dönemde de pozitif.
4. Tek çeyrek karşılaştırılabilir satış büyümesi pozitif.
5. Son 12 ay işletme faaliyetlerinden nakit akışı pozitif ve net kârın en az %80'i.
6. FAVÖK pozitif, net borç/FAVÖK < 2,5. Net nakit negatif net borçla ifade edilir.
7. Tamamlanan haftada fiyat > EMA5 > EMA22; RSI14 50–68 ve önceki haftaya göre yükseliyor.
8. Günlük tetik: kapanış önceki 20 seansın en yüksek fiyatının üzerinde; hacim önceki 20 seans ortalamasının en az 1,5 katı; kapanış gün aralığının üst %30'unda.
9. Günlük fiyat EMA20 + 2 ATR14 üzerinde ise `EXTENDED`; tetiklenen aday listesine alınmaz.

MACD histogram artışı, haftalık hacim, önceden Bollinger sıkışması, XU100'e göre güç ve yeni EMA kesişimi ek sıralama bonuslarıdır.
Bonus 0–50 arasıdır; yükseliş olasılığı değildir. XU100 eksikse o bonus verilmez.
Bankalar, sigortalar, holdingler, GYO'lar ve diğer finansallar ayrı modele ayrılır.
Bu eşikler başlangıç varsayımlarıdır; başarı oranı ölçülmemiştir.

## Bilanço girdi sözleşmesi

Şablon: `peg_momentum/templates/fundamentals.csv`. Veriler virgülle ayrılmış, ondalık noktasıyla yazılmış olmalı.

| Alan | Tanım |
|---|---|
| symbol | Örnek: THYAO; .IS eki yok |
| sector_model | nonfinancial; bank / insurance / holding / reit / other_financial ayrı model |
| source | Denetlenebilir kaynak URL veya belge kimliği |
| period_end | Son mali dönem bitişi, YYYY-MM-DD |
| published_at | Bilginin kullanılabilir olduğu ilk TAMAMLANMIŞ seans; kapanış sonrası açıklama ertesi seansa yazılır |
| valuation_date | F/K'nın ait olduğu kapanış tarihi |
| comparison_basis | Enflasyon, bölünme ve dönem karşılaştırma bazının açıklaması |
| comparable | true yalnızca karşılaştırılabilirlik gerçekten kontrol edildiyse |
| recurring_earnings_verified | true yalnızca tek seferlik kârlar kontrol edildiyse |
| pe_ttm | Son 12 ay F/K; EPS ile aynı kâr tanımı |
| eps_ttm, eps_prior_ttm | Son 12 ay ve bir yıl önceki 12 ay EPS, karşılaştırılabilir bazda |
| profit_ttm, profit_prior_ttm | Aynı dönemlerin ana ortaklık net kârı |
| operating_profit_q, operating_profit_prior_year_q | Son tek çeyrek ve geçen yıl aynı tek çeyrek esas faaliyet kârı |
| sales_q, sales_prior_year_q | Aynı tek çeyreklerin satışları |
| operating_cashflow_ttm | Son 12 ay işletme faaliyetlerinden nakit akışı |
| net_debt | Finansal borç eksi nakit |
| ebitda_ttm | Son 12 ay FAVÖK |

TMS29 tabloları aynı satın alma gücünde karşılaştırılmalı. Kümülatif 9 aylık tutar tek çeyrek diye kullanılamaz.
Yıllık/ara dönem tutarları farklı satın alma gücündeyse doğrudan çıkarılamaz.
F/K en fazla 7 gün, son mali dönem en fazla 200 gün eski kabul edilir.
Eksik alanlara 0/true yazılmaz. Yahoo `operatingIncome`, Türkçe esas faaliyet kârına doğrulamasız eşitlenmez.
Birden fazla yayın varsa tarama tarihinde bilinen son yayın kullanılır. Gelecek yayınlar kullanılmaz.
Bu denetim bayrakları bir veri doğrulama hizmetinin yerine geçmez; kaynağı sağlayan adaptör bunları kanıtlamalıdır.

Yerel CSV fiyatlarıyla çalışmak için:

```bash
python -m peg_momentum.engine --universe universe.csv --prices prices.csv --fundamentals fundamentals.csv --asof 2026-10-30 --output peg_momentum_output
```

Fiyat CSV kolonları: `symbol,date,open,high,low,close,volume`; evren CSV kolonu: `symbol`.
Her evren kodu raporlanır. En az 300 günlük ve 52 haftalık fiyat gerekir. Fiyat staleness ve yinelenen kayıtlar denetlenir;
eksik seans takviminin tam kontrolü henüz yoktur.

## Çıktılar ve veri kapsamı

- `all_results.csv`: her evren kodu için temel/teknik durum ve nedenler.
- `candidates.csv`: yalnızca doğrulanmış temel koşullarla TRIGGERED olanlar.
- `technical_preview.csv`: yalnızca fiyat grafiği değerlendirmesi; PEG aday listesi değildir.
- `download_diagnostics.csv`: her kod için indirilen satır sayısı/erişim hatası.
- `status.json`, `report.md`: kapsam ve veri eksikliği özeti.
- `prices.csv`, `universe.csv`: o çalıştırmada kullanılan girdiler.
- `source_financials.csv`: TRY cinsinden kaynak kalemler; birikimli/sonradan düzeltilmiş veriler olabilir, TTM değildir.
- `financial_source_diagnostics.csv`, `financial_source_status.json`: indirme kapsamı ve doğrulama sınırı.
- `kap_sample_audit.csv`: yalnızca desteklenen üç şirket için KAP kalem kontrolü; eşleşme PEG doğrulaması sayılmaz.

Otomatik runner'da yeterli fiyat veya değerlendirilmiş temel veri kapsamı %70 altındaysa aday listesi baskılanır,
`INCOMPLETE_DATA` yazılır. Hiçbir hisse şartları geçmediği halde veriler yeterliyse bu geçerli bir sıfır-aday sonucudur.
Temel kapsama PASS, REJECTED, SEPARATE_MODEL ve REVIEW_REQUIRED sayılır; DATA_MISSING sayılmaz.
Kapsamın %100 olmadığı her durumda eksikler dosyalarda görünür. 500+ kod listelemek her hissenin bütün verilerinin alındığı anlamına gelmez.

Durumlar: DATA_MISSING, SEPARATE_MODEL, REVIEW_REQUIRED, REJECTED, WAIT, RADAR, TRIGGERED, EXTENDED, COVERAGE_BLOCKED.
Eski çıktılar Telegram'a gönderilmez; bu modül hiçbir mesaj göndermez.

## Kaynak notları ve sınırlamalar

- yfinance fiyat parametreleri: https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html
- TMS29 karşılaştırma örneği: https://www.kap.org.tr/tr/api/file/download/4028328c9d7b22d2019ddf0b54bb705d
- İş Yatırım mali tablo baz notu: https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx?hisse=THYAO
- KAP özet/baz uyarısı: https://www.kap.org.tr/tr/sirket-finansal-bilgileri/4028e4a140f2ed720140f376bebb01a7
- Teknik eşikler yatırım performansı kanıtı değildir. Likidite/spread, işlem kısıtları, tavan/taban erişimi ve haber analizi modellenmedi.
- Gerçek geçmiş testte o tarihte bilinen bilanço arşivi, tarihteki evren, sonraki seans gerçekleştirilebilir giriş fiyatı,
  komisyon ve kayma gerekir. Bu değişiklik bunları uygulamaz.
