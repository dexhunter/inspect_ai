"""Milliseconds for one bootstrap_stderr evaluation.

Workload: 8,000 numeric sample scores and the default 1,000 resamples, one
metric pass over a large eval. Prints the timed call only; one warmup call
runs first and is not included.
"""

from __future__ import annotations

import math
import time

from inspect_ai.scorer._metric import SampleScore, Score
from inspect_ai.scorer._metrics.std import bootstrap_stderr


def _scores() -> list[SampleScore]:
    return [
        SampleScore(score=Score(value=float((i * 17) % 100) / 100.0))
        for i in range(8000)
    ]


def main() -> None:
    scores = _scores()
    metric = bootstrap_stderr()
    metric(scores)
    started = time.perf_counter()
    value = metric(scores)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not math.isfinite(value) or value < 0.0:
        raise SystemExit(f"bootstrap_stderr returned {value!r}")
    print(f"{elapsed_ms:.6f}")


if __name__ == "__main__":
    main()
