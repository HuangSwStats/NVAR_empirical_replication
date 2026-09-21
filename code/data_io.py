"""Read the packaged monthly levels; apply the documented full-sample transform."""
from pathlib import Path
import csv
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_panel():
    panels = []
    labels = None
    for name in ('trade_levels.csv', 'reer_levels.csv'):
        with (ROOT / 'data' / name).open(encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)
        dates = [r[0] for r in rows]
        countries = header[1:]
        if labels is not None:
            assert labels == (dates, countries)
        labels = dates, countries
        panels.append(np.array([[float(v) for v in r[1:]] for r in rows]).T)
    month = np.array([int(d[5:7]) for d in dates[1:]])
    transformed = []
    for levels in panels:
        changes = np.diff(levels, axis=1)
        seasonal = np.column_stack([changes[:, month == m].mean(1) for m in range(1, 13)])
        adjusted = changes - seasonal[:, month - 1]
        scale = adjusted.std(1, ddof=1, keepdims=True)
        assert np.isfinite(adjusted).all() and (scale > 0).all()
        transformed.append((adjusted - adjusted.mean(1, keepdims=True)) / scale)
    return transformed[0], transformed[1][:, None, :], countries, dates[1:]
