"""Result aggregation helpers."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def aggregate_runs(rows: Iterable[dict], group_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return pd.DataFrame(columns=[*group_columns, "AP_mean", "AP_std", "AF_mean", "AF_std", "n"])
    grouped = frame.groupby(group_columns, dropna=False)
    summary = grouped.agg(
        AP_mean=("AP", "mean"), AP_std=("AP", "std"),
        AF_mean=("AF", "mean"), AF_std=("AF", "std"), n=("AP", "count"),
    ).reset_index()
    for column in ("AP_std", "AF_std"):
        summary[column] = summary[column].fillna(0.0)
    return summary
