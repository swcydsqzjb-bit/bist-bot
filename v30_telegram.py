from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests


STATUS_FILE = Path("v30_status.json")
PATTERNS_FILE = Path("v30_learned_patterns.csv")
ADJUSTMENTS_FILE = Path("v30_feature_adjustments.csv")

TELEGRAM_API_URL = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 3900


def tx(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def sf(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        if path.stat().st_size == 0:
            return {}
    except OSError:
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8",
            )
        )
    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return {}


def load_csv(path: Path) -> pd.DataFrame:
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
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return tx(value).lower() in {
        "true",
        "1",
        "yes",
        "evet",
    }


def split_message(
    text: str,
    max_length: int = MAX_MESSAGE_LENGTH,
) -> list[str]:
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    current = ""

    for line in text.splitlines():
        candidate = (
            f"{current}\n{line}"
            if current
            else line
        )

        if len(candidate) <= max_length:
            current = candidate
            continue

        if current:
            parts.append(current)

        if len(line) <= max_length:
            current = line
        else:
            for index in range(
                0,
                len(line),
                max_length,
            ):
                parts.append(
                    line[
                        index:
                        index + max_length
                    ]
                )
            current = ""

    if current:
        parts.append(current)

    return parts


def send_telegram_message(
    token: str,
    chat_id: str,
    message: str,
) -> None:
    url = (
        f"{TELEGRAM_API_URL}/bot"
        f"{token}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": message,
            "disable_web_page_preview": True,
        },
        timeout=30,
    )

    if not response.ok:
        raise RuntimeError(
            "Telegram mesajı gönderilemedi: "
            f"{response.status_code} "
            f"{response.text}"
        )


def waiting_message(
    status: dict[str, Any],
) -> str:
    completed = int(
        sf(
            status.get(
                "completed_observation_count"
            )
        )
    )

    minimum = int(
        sf(
            status.get(
                "minimum_required"
            ),
            20,
        )
    )

    remaining = max(
        minimum - completed,
        0,
    )

    progress = (
        completed / minimum * 100.0
        if minimum > 0
        else 0.0
    )

    return (
        "🧠 LARUS V30 ÖRÜNTÜ ÖĞRENME\n\n"
        "Öğrenme durumu: VERİ TOPLANIYOR\n"
        f"Tamamlanan gözlem: {completed}/{minimum}\n"
        f"İlerleme: %{progress:.1f}\n"
        f"Kalan gözlem: {remaining}\n\n"
        "Aktif örüntü: 0\n"
        "Kullanılabilir bonus: 0.0 puan\n\n"
        "V29 sonuçları yeterli seviyeye "
        "ulaşınca öğrenme otomatik başlayacak.\n\n"
        "⚠️ Şu anda puanlara herhangi bir "
        "öğrenme bonusu uygulanmıyor."
    )


def active_learning_message(
    status: dict[str, Any],
    patterns: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> str:
    completed = int(
        sf(
            status.get(
                "completed_observation_count"
            )
        )
    )

    pattern_count = int(
        sf(
            status.get(
                "pattern_count"
            )
        )
    )

    active_pattern_count = int(
        sf(
            status.get(
                "active_pattern_count"
            )
        )
    )

    active_adjustment_count = int(
        sf(
            status.get(
                "active_adjustment_count"
            )
        )
    )

    total_bonus = sf(
        status.get(
            "total_available_bonus"
        )
    )

    strongest_feature = tx(
        status.get(
            "strongest_feature"
        )
    )

    lines = [
        "🧠 LARUS V30 ÖRÜNTÜ ÖĞRENME",
        "",
        "Öğrenme durumu: AKTİF",
        f"Tamamlanan gözlem: {completed}",
        f"Toplam örüntü: {pattern_count}",
        f"Aktif örüntü: {active_pattern_count}",
        (
            "Aktif özellik ayarı: "
            f"{active_adjustment_count}"
        ),
        (
            "Toplam kullanılabilir bonus: "
            f"+{total_bonus:.2f} puan"
        ),
    ]

    if strongest_feature:
        lines.extend(
            [
                (
                    "En güçlü özellik: "
                    f"{strongest_feature}"
                ),
            ]
        )

    active_patterns = pd.DataFrame()

    if not patterns.empty:
        if "active" in patterns.columns:
            active_patterns = patterns[
                patterns["active"].apply(
                    is_true
                )
            ].copy()
        else:
            active_patterns = patterns.copy()

    if not active_patterns.empty:
        sort_columns = [
            column
            for column in [
                "recommended_bonus",
                "success_rate",
                "sample_count",
            ]
            if column in active_patterns.columns
        ]

        if sort_columns:
            active_patterns = (
                active_patterns.sort_values(
                    sort_columns,
                    ascending=False,
                )
            )

        lines.extend(
            [
                "",
                "En güçlü öğrenilmiş örüntüler:",
            ]
        )

        for index, (_, row) in enumerate(
            active_patterns.head(5).iterrows(),
            start=1,
        ):
            feature = tx(
                row.get("feature")
            )

            direction = tx(
                row.get("direction")
            )

            source_decision = tx(
                row.get(
                    "source_decision"
                )
            )

            sample_count = int(
                sf(
                    row.get(
                        "sample_count"
                    )
                )
            )

            success_rate = sf(
                row.get(
                    "success_rate"
                )
            )

            bonus = sf(
                row.get(
                    "recommended_bonus"
                )
            )

            confidence = tx(
                row.get(
                    "confidence_class"
                )
            )

            lines.extend(
                [
                    "",
                    (
                        f"{index}. {feature} "
                        f"({direction})"
                    ),
                    (
                        "Kaynak karar: "
                        f"{source_decision}"
                    ),
                    (
                        "Örnek: "
                        f"{sample_count} | "
                        "Başarı: "
                        f"%{success_rate:.1f}"
                    ),
                    (
                        f"Bonus: +{bonus:.2f} | "
                        f"Güven: {confidence}"
                    ),
                ]
            )
    else:
        lines.extend(
            [
                "",
                "Henüz güvenilir aktif örüntü yok.",
            ]
        )

    active_adjustments = pd.DataFrame()

    if not adjustments.empty:
        if "active" in adjustments.columns:
            active_adjustments = adjustments[
                adjustments["active"].apply(
                    is_true
                )
            ].copy()
        else:
            active_adjustments = (
                adjustments.copy()
            )

    if not active_adjustments.empty:
        if (
            "normalized_bonus"
            in active_adjustments.columns
        ):
            active_adjustments = (
                active_adjustments.sort_values(
                    "normalized_bonus",
                    ascending=False,
                )
            )

        lines.extend(
            [
                "",
                "Özellik bonusları:",
            ]
        )

        for _, row in (
            active_adjustments.head(5).iterrows()
        ):
            feature = tx(
                row.get("feature")
            )

            direction = tx(
                row.get("direction")
            )

            bonus = sf(
                row.get(
                    "normalized_bonus"
                )
            )

            success_rate = sf(
                row.get(
                    "success_rate"
                )
            )

            lines.append(
                "• "
                f"{feature} / {direction}: "
                f"+{bonus:.2f} puan "
                f"(başarı %{success_rate:.1f})"
            )

    lines.extend(
        [
            "",
            "⚠️ V30 yalnızca istatistiksel "
            "öğrenme üretir.",
            (
                "Henüz V31 kurulmadığı için "
                "bu bonuslar ana karar "
                "puanlarına uygulanmıyor."
            ),
        ]
    )

    return "\n".join(lines)


def error_message(
    status: dict[str, Any],
) -> str:
    status_name = tx(
        status.get("status")
    )

    return (
        "⚠️ LARUS V30 ÖRÜNTÜ ÖĞRENME\n\n"
        "V30 raporu oluşturuldu ancak "
        "öğrenme verileri hazır değil.\n\n"
        f"Durum: {status_name or 'BİLİNMİYOR'}\n\n"
        "V29 sonuç dosyalarının oluştuğunu "
        "ve boş olmadığını kontrol et."
    )


def build_message() -> str:
    status = load_json(
        STATUS_FILE
    )

    patterns = load_csv(
        PATTERNS_FILE
    )

    adjustments = load_csv(
        ADJUSTMENTS_FILE
    )

    if not status:
        return (
            "⚠️ LARUS V30 ÖRÜNTÜ ÖĞRENME\n\n"
            "v30_status.json bulunamadı veya "
            "okunamadı."
        )

    status_name = tx(
        status.get("status")
    )

    learning_active = is_true(
        status.get(
            "learning_active"
        )
    )

    if status_name == (
        "insufficient_completed_observations"
    ):
        return waiting_message(
            status
        )

    if status_name == "ready":
        if learning_active:
            return active_learning_message(
                status,
                patterns,
                adjustments,
            )

        return (
            "🧠 LARUS V30 ÖRÜNTÜ ÖĞRENME\n\n"
            "Öğrenme motoru çalıştı ancak "
            "henüz güvenilir aktif örüntü "
            "oluşmadı.\n\n"
            "Tamamlanan gözlem: "
            f"{int(sf(status.get('completed_observation_count')))}\n"
            "Bulunan örüntü: "
            f"{int(sf(status.get('pattern_count')))}\n"
            "Aktif örüntü: "
            f"{int(sf(status.get('active_pattern_count')))}\n"
            "Kullanılabilir bonus: "
            f"{sf(status.get('total_available_bonus')):.2f}\n\n"
            "Veri biriktikçe sistem yeniden "
            "değerlendirme yapacak."
        )

    return error_message(
        status
    )


def main() -> None:
    token = tx(
        os.getenv("TOKEN")
    )

    chat_id = tx(
        os.getenv("CHAT_ID")
    )

    if not token:
        raise RuntimeError(
            "TOKEN environment değişkeni bulunamadı."
        )

    if not chat_id:
        raise RuntimeError(
            "CHAT_ID environment değişkeni bulunamadı."
        )

    message = build_message()

    message_parts = split_message(
        message
    )

    for part_number, part in enumerate(
        message_parts,
        start=1,
    ):
        send_telegram_message(
            token,
            chat_id,
            part,
        )

        print(
            "V30 Telegram mesajı gönderildi: "
            f"{part_number}/{len(message_parts)}"
        )


if __name__ == "__main__":
    main()
