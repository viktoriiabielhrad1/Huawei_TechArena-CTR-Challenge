#!/usr/bin/env python3
"""
TechArena CTR -- local time-based validation of the starter formula.

Uses validation_utils.py for the shared split/AUC logic. If you're
validating a different model (e.g. XGBoost), see validation_utils.py's
docstring -- same split pattern, just swap out the scoring section below
for your own model.

Usage:
    python3 local_validate.py --train train.csv
"""
import argparse
import math
import sys
from collections import defaultdict

from validation_utils import SPLIT_TS, SPLIT_UTC, read_rows, roc_auc

SMOOTHING_K = 50
CF_WEIGHT = 25.0


def logit(p, eps=1e-6):
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    a = ap.parse_args()

    item_clicks, item_impr = defaultdict(int), defaultdict(int)
    user_clicks, user_impr = defaultdict(int), defaultdict(int)
    user_interacted = defaultdict(set)
    total_clicks, total_rows = 0, 0
    holdout_rows = []

    print("pass 1/2: reading train.csv, splitting by timestamp ...", file=sys.stderr)
    for i, row in enumerate(read_rows(a.train)):
        if row["timestamp"] < SPLIT_TS:
            uid, iid, y = row["user_id"], row["item_id"], row["clicked"]
            item_clicks[iid] += y
            item_impr[iid] += 1
            user_clicks[uid] += y
            user_impr[uid] += 1
            total_clicks += y
            total_rows += 1
            if row["history_item_ids"]:
                user_interacted[uid].update(row["history_item_ids"])
            if y:
                user_interacted[uid].add(iid)
        else:
            holdout_rows.append(row)

        if (i + 1) % 2_000_000 == 0:
            print(f"  {i+1:,} rows", file=sys.stderr)

    print(f"fit rows: {total_rows:,}   holdout rows: {len(holdout_rows):,}", file=sys.stderr)

    global_ctr = total_clicks / total_rows
    item_ctr = {iid: (item_clicks[iid] + SMOOTHING_K * global_ctr) / (item_impr[iid] + SMOOTHING_K)
                for iid in item_impr}
    user_ctr = {uid: (user_clicks[uid] + SMOOTHING_K * global_ctr) / (user_impr[uid] + SMOOTHING_K)
                for uid in user_impr}
    g_logit = logit(global_ctr)

    print("building item-item co-click table (fit period only) ...", file=sys.stderr)
    item_pop = defaultdict(int)
    co = defaultdict(lambda: defaultdict(int))
    for uid, items in user_interacted.items():
        items = list(items)
        for it in items:
            item_pop[it] += 1
        for x in range(len(items)):
            for yix in range(x + 1, len(items)):
                a_, b_ = items[x], items[yix]
                co[a_][b_] += 1
                co[b_][a_] += 1
    item_pop_sqrt = {k: math.sqrt(v) for k, v in item_pop.items()}

    def cf_score(candidate_item, history_items):
        if candidate_item not in co or not history_items:
            return 0.0
        neighbors = co[candidate_item]
        total = 0.0
        denom_a = item_pop_sqrt.get(candidate_item, 0.0)
        for h in history_items:
            c = neighbors.get(h)
            if c:
                denom_b = item_pop_sqrt.get(h, 0.0)
                if denom_a and denom_b:
                    total += c / (denom_a * denom_b)
        return total

    def score_row(uid, iid, history_items):
        i_ctr = item_ctr.get(iid, global_ctr)
        u_ctr = user_ctr.get(uid, global_ctr)
        combo_logit = logit(i_ctr) + logit(u_ctr) - g_logit
        return combo_logit + CF_WEIGHT * cf_score(iid, history_items)

    print("pass 2/2: scoring the held-out rows ...", file=sys.stderr)
    pairs = []
    for row in holdout_rows:
        z = score_row(row["user_id"], row["item_id"], row["history_item_ids"])
        p = 1.0 / (1.0 + math.exp(-z))
        pairs.append((p, row["clicked"]))

    auc = roc_auc(pairs)
    print(f"local validation ROC AUC (fit on < {SPLIT_UTC.date()}, holdout >= {SPLIT_UTC.date()}): {auc:.4f}")


if __name__ == "__main__":
    main()