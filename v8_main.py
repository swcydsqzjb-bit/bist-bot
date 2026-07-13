from __future__ import annotations

import os

from v8_fusion import run_fusion
from v8_telegram import send_v8_report


def env_int(
    name: str,
    default: int = 0,
) -> int:
    try:
        return int(
            os.getenv(name, str(default))
        )

    except (TypeError, ValueError):
        return default


def main():
    symbol_limit = max(
        0,
        env_int("V8_SYMBOL_LIMIT", 0),
    )

    (
        fusion_df,
        candidates_df,
        quality_report,
    ) = run_fusion(
        symbol_limit=symbol_limit
    )

    total_symbols = (
        symbol_limit
        if symbol_limit > 0
        else len(quality_report)
    )

    valid_symbols = 0

    if (
        not quality_report.empty
        and "valid" in quality_report.columns
    ):
        valid_symbols = int(
            quality_report["valid"]
            .eq(True)
            .sum()
        )

    send_v8_report(
        candidates=candidates_df,
        total_symbols=total_symbols,
        valid_symbols=valid_symbols,
        shortlist_count=len(fusion_df),
    )


if __name__ == "__main__":
    main()
