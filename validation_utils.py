#!/usr/bin/env python3
"""
TechArena CTR -- shared helpers for local time-based validation.

Any approach (the starter formula, XGBoost, whatever comes next) can use
these to fit on earlier training days and validate on a later held-out day,
without spending one of our 10 daily real submissions.

Usage from your own script:

    from validation_utils import SPLIT_TS, read_rows, roc_auc

    # while reading train.csv, split each row yourself:
    for row in read_rows("train.csv"):
        if row["timestamp"] < SPLIT_TS:
            ... fit on this row (train your model / build your stats) ...
        else:
            ... hold this row out, score it with your model later ...

    # once you've scored all the holdout rows with your own model:
    auc = roc_auc(list_of_(predicted_probability, clicked)_tuples)
"""
import csv
import json
from datetime import datetime, timezone

# Fit on events before this UTC time; hold out events from this time
# (inclusive) up to the training cutoff (2025-01-27 00:00 UTC, where
# train.csv already ends).
SPLIT_UTC = datetime(2025, 1, 26, tzinfo=timezone.utc)
SPLIT_TS = int(SPLIT_UTC.timestamp())


def read_rows(path):
    """Yields dicts for every row of train.csv, with history_item_ids
    parsed from JSON and clicked/timestamp converted to int."""
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["history_item_ids"] = json.loads(row["history_item_ids"])
            row["clicked"] = int(row["clicked"])
            row["timestamp"] = int(row["timestamp"])
            yield row


def roc_auc(pairs):
    """pairs: list of (predicted_probability, clicked). Rank-based AUC
    (Mann-Whitney U), ties broken by average rank. No numpy/sklearn needed."""
    pairs = sorted(pairs, key=lambda t: t[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # average of the 1-indexed ranks i+1..j
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("holdout set needs at least one clicked and one non-clicked row")

    sum_ranks_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)