from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from v3_scanner import run_scanner
from v3_telegram import send_v3_report


ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


def count_valid_symbols(
    quality_report: pd.DataFrame,
) -> int:
    if (
        quality_report.empty
        or "valid" not in quality_report.columns
    ):
        return 0

    return int(
        quality_report["valid"]
        .astype(bool)
        .sum()
    )


def count_eligible_symbols(
    ranked_df: pd.DataFrame,
) -> int:
    if (
        ranked_df.empty
        or "eligible" not in ranked_df.columns
    ):
        return 0

    return int(
        ranked_df["eligible"]
        .astype(bool)
        .sum()
    )


def main():
    now = datetime.now(ISTANBUL_TZ)

    print(
        "V3 ana sistem başladı | "
        f"{now.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    ranked_df, candidates_df, quality_report = (
        run_scanner()
    )

    total_symbols = int(
        os.getenv("V3_SYMBOL_LIMIT", "0") or 0
    )

    if total_symbols <= 0:
        total_symbols = len(quality_report)

    valid_symbols = count_valid_symbols(
        quality_report
    )

    eligible_count = count_eligible_symbols(
        ranked_df
    )

    send_v3_report(
        candidates_df=candidates_df,
        total_symbols=total_symbols,
        valid_symbols=valid_symbols,
        eligible_count=eligible_count,
    )

    print("V3 ana sistem tamamlandı.")


if __name__ == "__main__":
    main()
