from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# GİRDİ VE ÇIKTI DOSYALARI
# =========================================================

V22_FILE = Path("v22_signal_states.csv")
V24_FILE = Path("v24_live_confirmations.csv")
V25_FILE = Path("v25_performance_evaluations.csv")
V26_FILE = Path("v26_optimized_portfolio.csv")

OUTPUT_FILE = Path("v27_master_decisions.csv")
STATUS_FILE = Path("v27_status.json")


# =========================================================
# ÇIKTI SÜTUNLARI
# =========================================================

OUTPUT_COLUMNS = [
    "v27_rank",
    "symbol",
    "v27_decision",
    "v27_master_score",
    "v27_reason",

    "optimized_weight_pct",
    "optimizer_score",
    "portfolio_role",

    "v22_signal_state",
    "v22_signal_score",

    "v24_state",
    "v24_score",

    "quality_score",
    "reliability_class",

    "consensus_score",
    "risk_class",
    "risk_score",

    "regime",
    "market_percentile",

    "best_horizon_days",
    "timing_confidence",

    "expected_return",
    "downside_20pct",
    "upside_80pct",

    "close",
    "reference_price",
]


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def sf(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Değeri güvenli biçimde float yapar.
    """

    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except (TypeError, ValueError):
        return default


def tx(value: Any) -> str:
    """
    Değeri güvenli biçimde metne dönüştürür.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_symbol(
    value: Any,
) -> str:
    """
    BIST sembolünü temizler.
    Örnek: POLHO.IS -> POLHO
    """

    symbol = tx(value).upper()

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


def load_csv(
    path: Path,
) -> pd.DataFrame:
    """
    CSV dosyasını güvenli biçimde okur.

    Dosya yoksa, boşsa veya okunamıyorsa
    boş DataFrame döndürür.
    """

    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                encoding="utf-8",
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        except Exception as exc:
            print(
                f"Uyarı: {path} UTF-8 olarak okunamadı: {exc}"
            )
            return pd.DataFrame()

    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return pd.DataFrame()


def normalize_symbol_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    DataFrame içindeki sembolleri temizler.
    """

    if frame.empty:
        return frame

    result = frame.copy()

    if "symbol" not in result.columns:
        return pd.DataFrame()

    result["symbol"] = result["symbol"].apply(
        normalize_symbol
    )

    result = result[
        result["symbol"].ne("")
    ].copy()

    result = result.drop_duplicates(
        subset=["symbol"],
        keep="first",
    )

    return result


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    """
    Eksik sütunu varsayılan değerle oluşturur.
    """

    if column not in frame.columns:
        frame[column] = default


# =========================================================
# CANLI DURUM ETKİSİ
# =========================================================

def live_state_bonus(
    state: str,
) -> float:
    """
    V24 canlı teyit durumunun ana skora etkisi.
    """

    normalized = tx(state).upper()

    bonuses = {
        "GÜÇLÜ CANLI TEYİT": 12.0,
        "CANLI TEYİT GELDİ": 10.0,
        "ERKEN TEYİT": 5.0,

        # Canlı teyit beklenmesi doğrudan büyük ceza değildir.
        "TEYİT BEKLE": -2.0,

        "PASİF": -7.0,
        "PASİF İZLEME": -7.0,

        "ŞİŞKİN / RİSKLİ": -18.0,
        "RİSKLİ - ELE": -20.0,
        "ELE": -20.0,
    }

    return bonuses.get(
        normalized,
        -3.0,
    )


# =========================================================
# V27 KARAR MOTORU
# =========================================================

def determine_decision(
    row: pd.Series,
) -> tuple[str, str]:
    """
    Bütün katmanları birlikte değerlendirerek
    nihai V27 kararını üretir.
    """

    master_score = sf(
        row.get("v27_master_score")
    )

    live_state = tx(
        row.get("v24_state")
    ).upper()

    v22_state = tx(
        row.get("v22_signal_state")
    ).upper()

    v22_score = sf(
        row.get("v22_signal_score")
    )

    risk_score = sf(
        row.get("risk_score"),
        100.0,
    )

    optimized_weight = sf(
        row.get("optimized_weight_pct")
    )

    optimizer_score = sf(
        row.get("optimizer_score")
    )

    quality_score = sf(
        row.get("quality_score"),
        50.0,
    )

    consensus_score = sf(
        row.get("consensus_score")
    )

    timing_confidence = sf(
        row.get("timing_confidence")
    )

    expected_return = sf(
        row.get("expected_return")
    )

    downside = sf(
        row.get("downside_20pct")
    )

    # -----------------------------------------------------
    # 1. KESİN ELEME ŞARTLARI
    # -----------------------------------------------------

    if live_state in {
        "ŞİŞKİN / RİSKLİ",
        "RİSKLİ - ELE",
        "ELE",
    }:
        return (
            "ELE",
            (
                "Canlı teknik görünümde şişkinlik "
                "veya belirgin risk tespit edildi"
            ),
        )

    if risk_score >= 65:
        return (
            "ELE",
            "Risk puanı kabul edilebilir seviyenin üzerinde",
        )

    if downside <= -6:
        return (
            "ELE",
            "Temkinli senaryoda aşağı yönlü risk yüksek",
        )

    if expected_return <= 0:
        return (
            "ELE",
            "Beklenen istatistiksel sonuç pozitif değil",
        )

    if quality_score < 30:
        return (
            "ELE",
            "Performans kalite görünümü yetersiz",
        )

    if v22_state in {
        "ELE",
        "RİSKLİ - ELE",
    }:
        return (
            "ELE",
            "V22 sinyal motoru adayı elemiş durumda",
        )

    # -----------------------------------------------------
    # 2. ÜST DÜZEY TEYİT
    # Gerçek canlı teyit şarttır.
    # -----------------------------------------------------

    if (
        live_state
        in {
            "CANLI TEYİT GELDİ",
            "GÜÇLÜ CANLI TEYİT",
        }
        and v22_state
        in {
            "GÜÇLÜ TEYİT",
            "İZLEMEYE AL",
            "TEYİT BEKLE",
        }
        and master_score >= 78
        and v22_score >= 72
        and consensus_score >= 72
        and risk_score <= 35
        and optimized_weight >= 8
    ):
        return (
            "ÜST DÜZEY TEYİT",
            (
                "Ana analiz katmanları, portföy motoru "
                "ve canlı teknik teyit aynı yönde güçlü"
            ),
        )

    # -----------------------------------------------------
    # 3. AKTİF İZLEME
    # Canlı teyit veya erken teyit oluşmuşsa.
    # -----------------------------------------------------

    if (
        live_state
        in {
            "CANLI TEYİT GELDİ",
            "GÜÇLÜ CANLI TEYİT",
            "ERKEN TEYİT",
        }
        and v22_state
        in {
            "GÜÇLÜ TEYİT",
            "İZLEMEYE AL",
            "TEYİT BEKLE",
        }
        and master_score >= 65
        and v22_score >= 62
        and risk_score <= 45
        and optimized_weight >= 8
    ):
        return (
            "AKTİF İZLEME",
            (
                "Canlı teknik teyit ile portföy "
                "uygunluğu birlikte oluştu"
            ),
        )

    # -----------------------------------------------------
    # 4. PORTFÖY MOTORUNUN SEÇTİĞİ GÜÇLÜ ADAY
    #
    # Canlı teyit henüz yoktur fakat:
    # - V22 çok güçlüdür,
    # - V26 anlamlı ağırlık ayırmıştır,
    # - risk düşüktür,
    # - motor uyumu yüksektir.
    #
    # Bu durumda aday pasife düşürülmez.
    # -----------------------------------------------------

    if (
        live_state == "TEYİT BEKLE"
        and v22_state
        in {
            "TEYİT BEKLE",
            "İZLEMEYE AL",
            "GÜÇLÜ TEYİT",
        }
        and optimized_weight >= 15
        and v22_score >= 75
        and consensus_score >= 75
        and risk_score <= 25
        and expected_return >= 2
    ):
        return (
            "AKTİF İZLEME",
            (
                "Portföy motoru güçlü ağırlık ayırdı; "
                "istatistiksel görünüm güçlü, canlı giriş "
                "teyidi bekleniyor"
            ),
        )

    # Biraz daha esnek aktif izleme şartı.
    if (
        live_state == "TEYİT BEKLE"
        and v22_state
        in {
            "TEYİT BEKLE",
            "İZLEMEYE AL",
        }
        and optimized_weight >= 12
        and optimizer_score >= 50
        and v22_score >= 70
        and consensus_score >= 70
        and risk_score <= 30
        and timing_confidence >= 70
        and expected_return >= 2
    ):
        return (
            "AKTİF İZLEME",
            (
                "Sinyal, zamanlama ve portföy motoru "
                "adayı birlikte destekliyor; canlı teyit "
                "tamamlanmadan kontrollü aktif izleme"
            ),
        )

    # -----------------------------------------------------
    # 5. TEYİT BEKLE
    # -----------------------------------------------------

    if live_state == "TEYİT BEKLE":
        if (
            v22_state
            in {
                "TEYİT BEKLE",
                "İZLEMEYE AL",
                "GÜÇLÜ TEYİT",
            }
            and v22_score >= 58
            and consensus_score >= 55
            and risk_score <= 55
            and expected_return > 0
        ):
            return (
                "TEYİT BEKLE",
                (
                    "Genel görünüm olumlu fakat canlı "
                    "teknik giriş teyidi henüz oluşmadı"
                ),
            )

        return (
            "PASİF İZLEME",
            (
                "Canlı teyit yok ve toplam görünüm "
                "aktif takip için yeterli değil"
            ),
        )

    # V24 farklı veya boş bir durum üretmiş olsa da
    # diğer katmanlar yeterince güçlüyse teyit beklenir.
    if (
        v22_state
        in {
            "TEYİT BEKLE",
            "İZLEMEYE AL",
        }
        and master_score >= 55
        and v22_score >= 58
        and risk_score <= 55
        and expected_return > 0
    ):
        return (
            "TEYİT BEKLE",
            (
                "Toplam görünüm olumlu fakat bütün "
                "teyit şartları tamamlanmadı"
            ),
        )

    # -----------------------------------------------------
    # 6. PASİF İZLEME
    # -----------------------------------------------------

    return (
        "PASİF İZLEME",
        (
            "Analiz katmanları ortak güçlü karar üretmedi"
        ),
    )


# =========================================================
# BOŞ SONUÇ YÖNETİMİ
# =========================================================

def save_empty_status(
    status_name: str,
) -> None:
    """
    Girdi olmadığında workflow'u durdurmadan
    başlıklı boş çıktı oluşturur.
    """

    pd.DataFrame(
        columns=OUTPUT_COLUMNS
    ).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "candidate_count": 0,
        "approved_count": 0,
        "top_level_confirmation_count": 0,
        "active_tracking_count": 0,
        "waiting_count": 0,
        "passive_count": 0,
        "eliminated_count": 0,
        "top_symbol": "",
        "top_decision": "",
        "top_score": 0.0,
        "version": "V27.2",
    }

    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


# =========================================================
# ANA ÇALIŞMA
# =========================================================

def main() -> None:
    signals = normalize_symbol_column(
        load_csv(V22_FILE)
    )

    live = normalize_symbol_column(
        load_csv(V24_FILE)
    )

    performance = normalize_symbol_column(
        load_csv(V25_FILE)
    )

    portfolio = normalize_symbol_column(
        load_csv(V26_FILE)
    )

    if signals.empty:
        save_empty_status(
            "v22_input_missing"
        )
        return

    merged = signals.copy()

    # -----------------------------------------------------
    # V24 CANLI TEYİT VERİLERİ
    # -----------------------------------------------------

    if not live.empty:
        live_columns = [
            column
            for column in [
                "symbol",
                "v24_state",
                "v24_score",
                "live_confirmation_score",
                "live_price",
            ]
            if column in live.columns
        ]

        live_data = live[
            live_columns
        ].copy()

        if (
            "v24_score" not in live_data.columns
            and "live_confirmation_score"
            in live_data.columns
        ):
            live_data["v24_score"] = (
                live_data[
                    "live_confirmation_score"
                ]
            )

        merged = merged.merge(
            live_data,
            on="symbol",
            how="left",
        )

    # -----------------------------------------------------
    # V25 PERFORMANS VERİLERİ
    # -----------------------------------------------------

    if not performance.empty:
        performance_columns = [
            column
            for column in [
                "symbol",
                "quality_score",
                "reliability_class",
            ]
            if column in performance.columns
        ]

        merged = merged.merge(
            performance[
                performance_columns
            ],
            on="symbol",
            how="left",
        )

    # -----------------------------------------------------
    # V26 PORTFÖY OPTİMİZASYON VERİLERİ
    # -----------------------------------------------------

    if not portfolio.empty:
        portfolio_columns = [
            column
            for column in [
                "symbol",
                "optimized_weight_pct",
                "optimizer_score",
                "portfolio_role",
            ]
            if column in portfolio.columns
        ]

        merged = merged.merge(
            portfolio[
                portfolio_columns
            ],
            on="symbol",
            how="left",
        )

    # -----------------------------------------------------
    # METİN SÜTUNLARI
    # -----------------------------------------------------

    text_defaults = {
        "v22_signal_state": "TEYİT BEKLE",
        "v24_state": "TEYİT BEKLE",
        "reliability_class": "YENİ",
        "risk_class": "ORTA",
        "regime": "",
        "portfolio_role": "",
    }

    for column, default in text_defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

        merged[column] = (
            merged[column]
            .fillna(default)
            .astype(str)
            .str.strip()
        )

    # -----------------------------------------------------
    # SAYISAL SÜTUNLAR
    #
    # V24 skoru veri yokken 0 değil 50 kabul edilir.
    # Böylece canlı teyit bekleyen aday gereksiz yere
    # aşırı cezalandırılmaz.
    # -----------------------------------------------------

    numeric_defaults = {
        "v22_signal_score": 0.0,
        "v24_score": 50.0,
        "quality_score": 50.0,

        "optimized_weight_pct": 0.0,
        "optimizer_score": 0.0,

        "consensus_score": 0.0,
        "risk_score": 100.0,

        "market_percentile": 0.0,
        "best_horizon_days": 1.0,
        "timing_confidence": 0.0,

        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,

        "close": 0.0,
        "reference_price": 0.0,
    }

    for column, default in numeric_defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(default)

    # V24 satırı var fakat v24_score boş veya sıfırsa,
    # TEYİT BEKLE için nötr skor kullan.
    waiting_mask = (
        merged["v24_state"]
        .str.upper()
        .eq("TEYİT BEKLE")
    )

    merged.loc[
        waiting_mask
        & (
            merged["v24_score"]
            <= 0
        ),
        "v24_score",
    ] = 50.0

    # Referans fiyat uyumluluğu.
    merged["close"] = np.where(
        merged["close"] > 0,
        merged["close"],
        merged["reference_price"],
    )

    merged["reference_price"] = np.where(
        merged["reference_price"] > 0,
        merged["reference_price"],
        merged["close"],
    )

    # -----------------------------------------------------
    # BEKLENEN GETİRİ BİLEŞENİ
    # -----------------------------------------------------

    expected_return_component = (
        np.clip(
            merged["expected_return"],
            -5,
            10,
        )
        + 5
    ) / 15 * 100

    expected_return_component = (
        expected_return_component.clip(
            0,
            100,
        )
    )

    # -----------------------------------------------------
    # V27 ANA SKOR
    #
    # V22 ve V26 etkisi artırıldı.
    # Canlı teyit yokken V24 nötr tutuldu.
    # Risk cezası aşırı olmayacak şekilde azaltıldı.
    # -----------------------------------------------------

    merged["v27_master_score"] = (
        merged["v22_signal_score"] * 0.28
        + merged["v24_score"] * 0.12
        + merged["optimizer_score"] * 0.20
        + merged["quality_score"] * 0.08
        + merged["consensus_score"] * 0.12
        + merged["timing_confidence"] * 0.08
        + merged["market_percentile"] * 0.05
        + expected_return_component * 0.05
        - merged["risk_score"] * 0.10
        + merged["v24_state"].apply(
            live_state_bonus
        )
    ).clip(
        0,
        100,
    )

    merged["v27_master_score"] = (
        merged["v27_master_score"]
        .round(2)
    )

    # -----------------------------------------------------
    # KARARLARI ÜRET
    # -----------------------------------------------------

    decisions = merged.apply(
        determine_decision,
        axis=1,
    )

    merged["v27_decision"] = [
        item[0]
        for item in decisions
    ]

    merged["v27_reason"] = [
        item[1]
        for item in decisions
    ]

    # -----------------------------------------------------
    # SIRALAMA
    # -----------------------------------------------------

    priority = {
        "ÜST DÜZEY TEYİT": 5,
        "AKTİF İZLEME": 4,
        "TEYİT BEKLE": 3,
        "PASİF İZLEME": 2,
        "ELE": 1,
    }

    merged["_priority"] = (
        merged["v27_decision"]
        .map(priority)
        .fillna(0)
    )

    merged = merged.sort_values(
        [
            "_priority",
            "v27_master_score",
            "optimized_weight_pct",
        ],
        ascending=False,
    ).drop(
        columns="_priority"
    ).reset_index(
        drop=True
    )

    merged.insert(
        0,
        "v27_rank",
        range(
            1,
            len(merged) + 1,
        ),
    )

    # -----------------------------------------------------
    # ÇIKTI DOSYASI
    # -----------------------------------------------------

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in merged.columns:
            result[column] = merged[column]
        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------
    # DURUM DOSYASI
    # -----------------------------------------------------

    approved_states = {
        "ÜST DÜZEY TEYİT",
        "AKTİF İZLEME",
    }

    status = {
        "status": "ready",

        "candidate_count": int(
            len(result)
        ),

        "approved_count": int(
            result["v27_decision"]
            .isin(approved_states)
            .sum()
        ),

        "top_level_confirmation_count": int(
            (
                result["v27_decision"]
                == "ÜST DÜZEY TEYİT"
            ).sum()
        ),

        "active_tracking_count": int(
            (
                result["v27_decision"]
                == "AKTİF İZLEME"
            ).sum()
        ),

        "waiting_count": int(
            (
                result["v27_decision"]
                == "TEYİT BEKLE"
            ).sum()
        ),

        "passive_count": int(
            (
                result["v27_decision"]
                == "PASİF İZLEME"
            ).sum()
        ),

        "eliminated_count": int(
            (
                result["v27_decision"]
                == "ELE"
            ).sum()
        ),

        "top_symbol": (
            tx(
                result.iloc[0]["symbol"]
            )
            if len(result)
            else ""
        ),

        "top_decision": (
            tx(
                result.iloc[0]["v27_decision"]
            )
            if len(result)
            else ""
        ),

        "top_score": (
            round(
                sf(
                    result.iloc[0][
                        "v27_master_score"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),

        "version": "V27.2",
    }

    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
