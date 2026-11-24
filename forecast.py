#!/usr/bin/env python3
"""Бэктест базовых прогнозов на временном ряду: MAE и MAPE по каждому методу.

usage: forecast.py backtest --input series.csv --horizon 7 [--min-train 56]
                            [--out metrics.json] [--csv metrics.csv] [--json]
Коды: 0 — готово, 2 — аргументы, 3 — нет файла, 4 — ошибка данных.
"""
import argparse
import csv
import json
import os
import sys

METHODS = ("naive", "mean", "ma7", "seasonal7", "trend")


def read_series(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "date" not in fields or "value" not in fields:
            print(f"нужны колонки date и value, есть: {', '.join(fields)}", file=sys.stderr)
            return None
        series = []
        for line, row in enumerate(reader, start=2):
            raw = (row.get("value") or "").replace(" ", "").replace("\u00a0", "").replace(",", ".")
            try:
                series.append((row.get("date", ""), float(raw)))
            except ValueError:
                print(f"строка {line}: не число: {row.get('value')!r}", file=sys.stderr)
                return None
    return series


def moving_average(history, window):
    tail = history[-window:]
    return sum(tail) / len(tail)


def trend_forecast(history, steps):
    n = len(history)
    if n < 2:
        return [history[-1]] * steps
    x_mean = (n - 1) / 2
    y_mean = sum(history) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(history))
    den = sum((i - x_mean) ** 2 for i in range(n))
    slope = num / den if den else 0.0
    return [y_mean + slope * ((n - 1 + step) - x_mean) for step in range(1, steps + 1)]


def forecast(method, history, steps):
    if not history:
        raise ValueError("пустая история")
    if method == "naive":
        return [history[-1]] * steps
    if method == "mean":
        return [sum(history) / len(history)] * steps
    if method == "ma7":
        return [moving_average(history, 7)] * steps
    if method == "seasonal7":
        return [history[-7 + (step % 7)] for step in range(steps)]
    if method == "trend":
        return trend_forecast(history, steps)
    raise ValueError(f"неизвестный метод: {method}")


def errors(actual, predicted):
    pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    if not pairs:
        return None
    mae = sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)
    mape = sum(abs(a - p) / abs(a) for a, p in pairs) / len(pairs) * 100
    return mae, mape


def backtest(values, horizon, min_train):
    if len(values) <= min_train + horizon - 1:
        raise ValueError(f"ряда мало: {len(values)} точек при horizon={horizon} "
                         f"и min_train={min_train}")
    result = {method: {"errors": [], "origins": 0} for method in METHODS}
    for origin in range(min_train, len(values) - horizon + 1):
        history = values[:origin]
        actual = values[origin:origin + horizon]
        for method in METHODS:
            stats = errors(actual, forecast(method, history, horizon))
            if stats is None:
                continue
            result[method]["errors"].append(stats)
            result[method]["origins"] += 1
    summary = {}
    for method, data in result.items():
        if not data["errors"]:
            summary[method] = {"mae": None, "mape": None, "origins": 0}
            continue
        mae = sum(e[0] for e in data["errors"]) / len(data["errors"])
        mape = sum(e[1] for e in data["errors"]) / len(data["errors"])
        summary[method] = {"mae": round(mae, 2), "mape": round(mape, 2),
                           "origins": data["origins"]}
    best = min((m for m in summary if summary[m]["mae"] is not None),
               key=lambda m: summary[m]["mae"])
    return {"rows": len(values), "horizon": horizon, "min_train": min_train,
            "origins": summary[best]["origins"], "best": best, "methods": summary}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    back = sub.add_parser("backtest", help="прогнать методы по истории с окном horizon")
    back.add_argument("--input", required=True)
    back.add_argument("--horizon", type=int, default=7)
    back.add_argument("--min-train", type=int, default=56)
    back.add_argument("--out")
    back.add_argument("--csv")
    back.add_argument("--json", action="store_true")
    back.set_defaults(func=run_backtest)
    return parser


def run_backtest(args):
    if args.horizon < 1 or args.min_train < 7:
        print("horizon >= 1 и min_train >= 7", file=sys.stderr)
        return 2
    if not os.path.exists(args.input):
        print(f"нет файла: {args.input}", file=sys.stderr)
        return 3
    series = read_series(args.input)
    if series is None:
        return 4
    if not series:
        print("ряд пустой", file=sys.stderr)
        return 4
    try:
        report = backtest([value for _, value in series], args.horizon, args.min_train)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    report["input"] = args.input
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if args.csv:
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["method", "mae", "mape", "origins"])
            for method, stats in report["methods"].items():
                writer.writerow([method, stats["mae"], stats["mape"], stats["origins"]])
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print(f"ряд: {report['rows']} точек, горизонт {report['horizon']}, "
              f"окон {report['origins']}, лучший метод: {report['best']}")
        for method, stats in report["methods"].items():
            print(f"  {method:<10} MAE {stats['mae']:>8}  MAPE {stats['mape']:>7}%")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
