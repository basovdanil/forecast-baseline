"""Генератор демо-ряда: 730 дней дневной выручки магазина, сид 42.

python make_demo.py [--out demo/sales.csv]
"""
import argparse
import csv
import datetime as dt
import math
import os
import random


def build(days=730, seed=42):
    rng = random.Random(seed)
    start = dt.date(2025, 1, 1)
    rows = []
    for day in range(days):
        date = start + dt.timedelta(days=day)
        trend = 42000 + day * 18
        weekday = 1.28 if date.weekday() >= 5 else 1.0
        monthly = 1 + 0.07 * math.sin(2 * math.pi * day / 30.4)
        noise = rng.gauss(0, 0.06)
        value = round(trend * weekday * monthly * (1 + noise), 2)
        rows.append((date.isoformat(), value))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("demo", "sales.csv"))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = build(args.days, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "value"])
        writer.writerows(rows)
    print(f"{args.out}: {len(rows)} строк, {rows[0][0]} .. {rows[-1][0]}")


if __name__ == "__main__":
    main()
