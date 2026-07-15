from __future__ import annotations

from typing import Iterable

import numpy as np


def clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def calculate_consensus(
    scores: Iterable[float],
) -> tuple[float, float, int]:
    """
    Returns:
      consensus_score: 0-100, higher means engines agree.
      dispersion: standard deviation of valid engine scores.
      valid_count: number of scores used.
    """
    values = np.array(
        [
            float(score)
            for score in scores
            if score is not None
            and np.isfinite(float(score))
        ],
        dtype=float,
    )

    if len(values) == 0:
        return 0.0, 0.0, 0

    if len(values) == 1:
        return 45.0, 0.0, 1

    dispersion = float(np.std(values))

    # 0 std => 100 consensus.
    # Around 25 std => near 0 consensus.
    consensus = 100.0 - dispersion * 4.0

    # Penalize missing engine coverage.
    coverage_factor = min(1.0, len(values) / 5.0)
    consensus *= 0.70 + coverage_factor * 0.30

    return (
        round(clamp(consensus), 2),
        round(dispersion, 2),
        int(len(values)),
    )
