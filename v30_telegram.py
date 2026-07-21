from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================
# DOSYALAR
# ============================================================

V28_STATUS_FILE = Path("v28_status.json")
V28_HISTORY_FILE = Path("v28_observation_history.csv")
V28_LATEST_FILE = Path("v28_latest_observations.csv")

V29_STATUS_FILE = Path("v29_status.json")
V29_EVALUATED_FILE = Path("v29_evaluated_observations.csv")

V30_STATUS_FILE = Path("v30_status.json")
V30_PATTERNS_FILE = Path("v30_learned_patterns.csv")
V30_ADJUSTMENTS_FILE = Path("v30_feature_adjustments.csv")

TELEGRAM_STATE_FILE = Path("v30_telegram_state.json")

TELEGRAM_API_URL = "https://api.telegram.org"
MAX_MESSAGE_LENGTH = 3900


# ============================================================
# TEMEL YARDIMCI FONKSIYONLAR
# ============================================================

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


def si(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    return tx(value).lower() in {
        "true",
        "1",
        "yes",
        "evet",
        "aktif",
        "active",
    }


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
            path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return {}


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            continue

    print(
        f"Uyarı: {path} okunamadı."
    )
    return pd.DataFrame()


def count_rows(path: Path) -> int:
    dataframe = load_csv(path)

    if dataframe.empty:
        return 0

    return int(len(dataframe))


def positive_delta(
    current: int,
    previous: int,
) -> int:
    return max(
        current - previous,
        0,
    )


# ============================================================
# SAYIM FONKSIYONLARI
# ============================================================

def get_v28_total_observations(
    status: dict[str, Any],
) -> int:
    possible_keys = [
        "history_count",
        "observation_count",
        "total_observation_count",
        "total_count",
        "pool_count",
    ]

    for key in possible_keys:
        if key in status:
            return si(
                status.get(key)
            )

    return count_rows(
        V28_HISTORY_FILE
    )


def get_v28_latest_count(
    status: dict[str, Any],
) -> int:
    possible_keys = [
        "latest_observation_count",
        "new_observation_count",
        "today_observation_count",
        "tracked_count",
    ]

    for key in possible_keys:
        if key in status:
            return si(
                status.get(key)
            )

    return count_rows(
        V28_LATEST_FILE
    )


def get_completed_observation_count(
    v29_status: dict[str, Any],
    v30_status: dict[str, Any],
) -> int:
    if (
        "completed_observation_count"
        in v30_status
    ):
        return si(
            v30_status.get(
                "completed_observation_count"
            )
        )

    possible_keys = [
        "completed_observation_count",
        "completed_count",
        "successful_observation_count",
    ]

    for key in possible_keys:
        if key in v29_status:
            return si(
                v29_status.get(key)
            )

    evaluated = load_csv(
        V29_EVALUATED_FILE
    )

    if evaluated.empty:
        return 0

    completion_columns = [
        "completed",
        "is_completed",
        "evaluation_completed",
        "outcome_completed",
    ]

    for column in completion_columns:
        if column in evaluated.columns:
            return int(
                evaluated[column]
                .apply(is_true)
                .sum()
            )

    horizon_columns = [
        "return_1d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_15d",
    ]

    existing_horizons = [
        column
        for column in horizon_columns
        if column in evaluated.columns
    ]

    if existing_horizons:
        completed_mask = (
            evaluated[existing_horizons]
            .notna()
            .any(axis=1)
        )

        return int(
            completed_mask.sum()
        )

    return 0


def get_pattern_count(
    status: dict[str, Any],
    patterns: pd.DataFrame,
) -> int:
    if "pattern_count" in status:
        return si(
            status.get("pattern_count")
        )

    return int(len(patterns))


def get_active_pattern_count(
    status: dict[str, Any],
    patterns: pd.DataFrame,
) -> int:
    if "active_pattern_count" in status:
        return si(
            status.get(
                "active_pattern_count"
            )
        )

    if patterns.empty:
        return 0

    if "active" in patterns.columns:
        return int(
            patterns["active"]
            .apply(is_true)
            .sum()
        )

    return int(len(patterns))


def get_active_adjustment_count(
    status: dict[str, Any],
    adjustments: pd.DataFrame,
) -> int:
    if (
        "active_adjustment_count"
        in status
    ):
        return si(
            status.get(
                "active_adjustment_count"
            )
        )

    if adjustments.empty:
        return 0

    if "active" in adjustments.columns:
        return int(
            adjustments["active"]
            .apply(is_true)
            .sum()
        )

    return int(len(adjustments))


# ============================================================
# GUNCEL DURUM VE ONCEKI DURUM
# ============================================================

def build_current_snapshot(
    v28_status: dict[str, Any],
    v29_status: dict[str, Any],
    v30_status: dict[str, Any],
    patterns: pd.DataFrame,
    adjustments: pd.DataFrame,
) -> dict[str, Any]:
    total_observations = (
        get_v28_total_observations(
            v28_status
        )
    )

    latest_observation_count = (
        get_v28_latest_count(
            v28_status
        )
    )

    completed_observations = (
        get_completed_observation_count(
            v29_status,
            v30_status,
        )
    )

    pattern_count = get_pattern_count(
        v30_status,
        patterns,
    )

    active_pattern_count = (
        get_active_pattern_count(
            v30_status,
            patterns,
        )
    )

    active_adjustment_count = (
        get_active_adjustment_count(
            v30_status,
            adjustments,
        )
    )

    return {
        "updated_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "total_observations": (
            total_observations
        ),
        "latest_observation_count": (
            latest_observation_count
        ),
        "completed_observations": (
            completed_observations
        ),
        "pattern_count": pattern_count,
        "active_pattern_count": (
            active_pattern_count
        ),
        "active_adjustment_count": (
            active_adjustment_count
        ),
        "total_available_bonus": sf(
            v30_status.get(
                "total_available_bonus"
            )
        ),
    }


def build_daily_changes(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    first_measurement = not bool(
        previous
    )

    if first_measurement:
        return {
            "first_measurement": True,
            "new_observations": 0,
            "new_completed_observations": 0,
            "new_patterns": 0,
            "new_active_patterns": 0,
            "new_active_adjustments": 0,
        }

    return {
        "first_measurement": False,
        "new_observations": positive_delta(
            si(
                current.get(
                    "total_observations"
                )
            ),
            si(
                previous.get(
                    "total_observations"
                )
            ),
        ),
        "new_completed_observations": (
            positive_delta(
                si(
                    current.get(
                        "completed_observations"
                    )
                ),
                si(
                    previous.get(
                        "completed_observations"
                    )
                ),
            )
        ),
        "new_patterns": positive_delta(
            si(
                current.get(
                    "pattern_count"
                )
            ),
            si(
                previous.get(
                    "pattern_count"
                )
            ),
        ),
        "new_active_patterns": (
            positive_delta(
                si(
                    current.get(
                        "active_pattern_count"
                    )
                ),
                si(
                    previous.get(
                        "active_pattern_count"
                    )
                ),
            )
        ),
        "new_active_adjustments": (
            positive_delta(
                si(
                    current.get(
                        "active_adjustment_count"
                    )
                ),
                si(
                    previous.get(
                        "active_adjustment_count"
                    )
                ),
            )
        ),
    }


# ============================================================
# TELEGRAM
# ============================================================

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


# ============================================================
# MESAJ BOLUMLERI
# ============================================================

def daily_progress_lines(
    current: dict[str, Any],
    changes: dict[str, Any],
) -> list[str]:
    total_observations = si(
        current.get(
            "total_observations"
        )
    )

    completed = si(
        current.get(
            "completed_observations"
        )
    )

    latest_count = si(
        current.get(
            "latest_observation_count"
        )
    )

    lines = [
        "📊 GÜNLÜK İLERLEME",
    ]

    if changes.get(
        "first_measurement"
    ):
        lines.extend(
            [
                "• İlk takip ölçümü oluşturuldu.",
                (
                    "• Mevcut gözlem havuzu: "
                    f"{total_observations}"
                ),
                (
                    "• Bu çalıştırmadaki aday: "
                    f"{latest_count}"
                ),
                (
                    "• Tamamlanan gözlem: "
                    f"{completed}"
                ),
            ]
        )
    else:
        lines.extend(
            [
                (
                    "• Yeni gözlem eklendi: "
                    f"+{si(changes.get('new_observations'))}"
                ),
                (
                    "• Yeni tamamlanan gözlem: "
                    f"+{si(changes.get('new_completed_observations'))}"
                ),
                (
                    "• Toplam gözlem havuzu: "
                    f"{total_observations}"
                ),
                (
                    "• Bu çalıştırmadaki aday: "
                    f"{latest_count}"
                ),
            ]
        )

    return lines


def waiting_message(
    status: dict[str, Any],
    current: dict[str, Any],
    changes: dict[str, Any],
) -> str:
    completed = si(
        status.get(
            "completed_observation_count"
        ),
        si(
            current.get(
                "completed_observations"
            )
        ),
    )

    minimum = si(
        status.get(
            "minimum_required"
        ),
        20,
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

    lines = [
        "🧠 LARUS V30 ÖRÜNTÜ ÖĞRENME",
        "",
        "Öğrenme durumu: VERİ TOPLANIYOR",
        "",
    ]

    lines.extend(
        daily_progress_lines(
            current,
            changes,
        )
    )

    lines.extend(
        [
            "",
            "📈 GENEL DURUM",
            (
                "• Tamamlanan gözlem: "
                f"{completed}/{minimum}"
            ),
            (
                "• İlerleme: "
                f"%{progress:.1f}"
            ),
            (
                "• Kalan gözlem: "
                f"{remaining}"
            ),
            "",
            "🎯 ÖĞRENME DURUMU",
            "• Aktif örüntü: 0",
            "• Kullanılabilir bonus: 0.0 puan",
            "",
            (
                "V29 sonuçları yeterli seviyeye "
                "ulaşınca öğrenme otomatik "
                "başlayacak."
            ),
            "",
            (
                "⚠️ Şu anda puanlara herhangi "
                "bir öğrenme bonusu uygulanmıyor."
            ),
        ]
    )

    return "\n".join(lines)


def sort_active_patterns(
    patterns: pd.DataFrame,
) -> pd.DataFrame:
    if patterns.empty:
        return pd.DataFrame()

    active_patterns = patterns.copy()

    if "active" in active_patterns.columns:
        active_patterns = active_patterns[
            active_patterns["active"]
            .apply(is_true)
        ].copy()

    sort_columns = [
        column
        for column in [
            "recommended_bonus",
            "normalized_bonus",
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

    return active_patterns


def active_learning_message(
    status: dict[str, Any],
    patterns: pd.DataFrame,
    adjustments: pd.DataFrame,
    current: dict[str, Any],
    changes: dict[str, Any],
) -> str:
    completed = si(
        status.get(
            "completed_observation_count"
        ),
        si(
            current.get(
                "completed_observations"
            )
        ),
    )

    pattern_count = si(
        status.get("pattern_count"),
        si(
            current.get(
                "pattern_count"
            )
        ),
    )

    active_pattern_count = si(
        status.get(
            "active_pattern_count"
        ),
        si(
            current.get(
                "active_pattern_count"
            )
        ),
    )

    active_adjustment_count = si(
        status.get(
            "active_adjustment_count"
        ),
        si(
            current.get(
                "active_adjustment_count"
            )
        ),
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
        "🧠 LARUS V30 ÖĞRENME AKTİF",
        "",
    ]

    lines.extend(
        daily_progress_lines(
            current,
            changes,
        )
    )

    lines.extend(
        [
            "",
            "📈 ÖĞRENME ÖZETİ",
            (
                "• Tamamlanan gözlem: "
                f"{completed}"
            ),
            (
                "• Bugün öğrenilen örüntü: "
                f"+{si(changes.get('new_patterns'))}"
            ),
            (
                "• Toplam örüntü: "
                f"{pattern_count}"
            ),
            (
                "• Aktif örüntü: "
                f"{active_pattern_count}"
            ),
            (
                "• Yeni aktif örüntü: "
                f"+{si(changes.get('new_active_patterns'))}"
            ),
            (
                "• Aktif özellik ayarı: "
                f"{active_adjustment_count}"
            ),
            (
                "• Kullanılabilir bonus: "
                f"+{total_bonus:.2f} puan"
            ),
        ]
    )

    if strongest_feature:
        lines.append(
            "• En güçlü özellik: "
            f"{strongest_feature}"
        )

    active_patterns = (
        sort_active_patterns(
            patterns
        )
    )

    if not active_patterns.empty:
        lines.extend(
            [
                "",
                "🏆 EN GÜÇLÜ ÖRÜNTÜLER",
            ]
        )

        for index, (_, row) in enumerate(
            active_patterns.head(5).iterrows(),
            start=1,
        ):
            feature = (
                tx(row.get("feature"))
                or tx(
                    row.get(
                        "pattern_name"
                    )
                )
                or "Bilinmeyen örüntü"
            )

            direction = tx(
                row.get("direction")
            )

            sample_count = si(
                row.get(
                    "sample_count"
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
                ),
                sf(
                    row.get(
                        "normalized_bonus"
                    )
                ),
            )

            confidence = tx(
                row.get(
                    "confidence_class"
                )
            )

            title = feature

            if direction:
                title += (
                    f" / {direction}"
                )

            lines.extend(
                [
                    "",
                    f"{index}. {title}",
                    (
                        "   Örnek: "
                        f"{sample_count} | "
                        "Başarı: "
                        f"%{success_rate:.1f}"
                    ),
                    (
                        "   Bonus: "
                        f"+{bonus:.2f}"
                        + (
                            f" | Güven: {confidence}"
                            if confidence
                            else ""
                        )
                    ),
                ]
            )
    else:
        lines.extend(
            [
                "",
                (
                    "Henüz güvenilir aktif "
                    "örüntü oluşmadı."
                ),
            ]
        )

    lines.extend(
        [
            "",
            (
                "⚠️ V31 henüz kurulmadığı için "
                "öğrenilen bonuslar ana karar "
                "puanlarına uygulanmıyor."
            ),
        ]
    )

    return "\n".join(lines)


def ready_without_active_learning_message(
    status: dict[str, Any],
    current: dict[str, Any],
    changes: dict[str, Any],
) -> str:
    completed = si(
        status.get(
            "completed_observation_count"
        ),
        si(
            current.get(
                "completed_observations"
            )
        ),
    )

    pattern_count = si(
        status.get("pattern_count"),
        si(
            current.get(
                "pattern_count"
            )
        ),
    )

    active_pattern_count = si(
        status.get(
            "active_pattern_count"
        ),
        si(
            current.get(
                "active_pattern_count"
            )
        ),
    )

    total_bonus = sf(
        status.get(
            "total_available_bonus"
        )
    )

    lines = [
        "🧠 LARUS V30 ÖRÜNTÜ ÖĞRENME",
        "",
        (
            "Öğrenme motoru çalıştı ancak "
            "henüz güvenilir aktif örüntü "
            "oluşmadı."
        ),
        "",
    ]

    lines.extend(
        daily_progress_lines(
            current,
            changes,
        )
    )

    lines.extend(
        [
            "",
            "📈 GENEL DURUM",
            (
                "• Tamamlanan gözlem: "
                f"{completed}"
            ),
            (
                "• Toplam örüntü: "
                f"{pattern_count}"
            ),
            (
                "• Aktif örüntü: "
                f"{active_pattern_count}"
            ),
            (
                "• Kullanılabilir bonus: "
                f"{total_bonus:.2f} puan"
            ),
            "",
            (
                "Veri biriktikçe sistem "
                "yeniden değerlendirme yapacak."
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
        "V28, V29 ve V30 sonuç dosyalarını "
        "kontrol et."
    )


# ============================================================
# RAPOR OLUSTURMA
# ============================================================

def build_report() -> tuple[
    str,
    dict[str, Any],
]:
    v28_status = load_json(
        V28_STATUS_FILE
    )

    v29_status = load_json(
        V29_STATUS_FILE
    )

    v30_status = load_json(
        V30_STATUS_FILE
    )

    patterns = load_csv(
        V30_PATTERNS_FILE
    )

    adjustments = load_csv(
        V30_ADJUSTMENTS_FILE
    )

    previous_snapshot = load_json(
        TELEGRAM_STATE_FILE
    )

    current_snapshot = (
        build_current_snapshot(
            v28_status,
            v29_status,
            v30_status,
            patterns,
            adjustments,
        )
    )

    changes = build_daily_changes(
        current_snapshot,
        previous_snapshot,
    )

    if not v30_status:
        message = (
            "⚠️ LARUS V30 ÖRÜNTÜ ÖĞRENME\n\n"
            "v30_status.json bulunamadı veya "
            "okunamadı."
        )

        return (
            message,
            current_snapshot,
        )

    status_name = tx(
        v30_status.get("status")
    )

    learning_active = is_true(
        v30_status.get(
            "learning_active"
        )
    )

    if status_name == (
        "insufficient_completed_observations"
    ):
        message = waiting_message(
            v30_status,
            current_snapshot,
            changes,
        )

        return (
            message,
            current_snapshot,
        )

    if (
        status_name == "ready"
        and learning_active
    ):
        message = active_learning_message(
            v30_status,
            patterns,
            adjustments,
            current_snapshot,
            changes,
        )

        return (
            message,
            current_snapshot,
        )

    if status_name == "ready":
        message = (
            ready_without_active_learning_message(
                v30_status,
                current_snapshot,
                changes,
            )
        )

        return (
            message,
            current_snapshot,
        )

    return (
        error_message(v30_status),
        current_snapshot,
    )


# ============================================================
# ANA CALISMA
# ============================================================

def main() -> None:
    token = tx(
        os.getenv("TOKEN")
    )

    chat_id = tx(
        os.getenv("CHAT_ID")
    )

    if not token:
        raise RuntimeError(
            "TOKEN environment değişkeni "
            "bulunamadı."
        )

    if not chat_id:
        raise RuntimeError(
            "CHAT_ID environment değişkeni "
            "bulunamadı."
        )

    message, current_snapshot = (
        build_report()
    )

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
            f"{part_number}/"
            f"{len(message_parts)}"
        )

    # Durum yalnızca Telegram mesajı başarıyla
    # gönderildikten sonra kaydedilir.
    save_json(
        TELEGRAM_STATE_FILE,
        current_snapshot,
    )

    print(
        "V30 Telegram takip durumu "
        f"{TELEGRAM_STATE_FILE} dosyasına "
        "kaydedildi."
    )


if __name__ == "__main__":
    main()
