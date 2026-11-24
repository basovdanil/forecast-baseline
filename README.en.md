# forecast-baseline

[![tests](https://github.com/basovdanil/forecast-baseline/actions/workflows/tests.yml/badge.svg)](https://github.com/basovdanil/forecast-baseline/actions/workflows/tests.yml)

Backtest of five baseline forecasts on a time series with rolling windows: naive, mean, moving
average, weekly seasonality and a least-squares trend. Reports MAE and MAPE per method, so a model
has to beat the naive one to be worth anything.

Standard library only, Python 3.8+. No numpy, no statsmodels.

## Quick start

```
python forecast.py backtest --input demo/sales.csv --horizon 7 --min-train 56
```

On the demo series (730 days, 668 windows) the best baseline is weekly seasonality:
MAE 4 616,03 and **MAPE 8,81 %**, against 14,92 % for naive — the point being that a seasonal
baseline is hard to beat and any model should be compared with it first.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | done |
| `2` | bad arguments |
| `3` | file not found |
| `4` | data error: no `date`/`value` columns, non-numeric value, series shorter than the window |

## Tests

```
python -m unittest discover -s tests
```

16 tests: each method on a short series, weekly shift, least squares on a perfect line, MAE/MAPE
with zero actuals, window count, number parsing with spaces and decimal comma, exit codes, `--out`
and `--csv` output.

## Scope

Baselines only — no boosting, no intervals. The series is synthetic.

---

More projects: [github.com/basovdanil](https://github.com/basovdanil) · write-ups: [portfolio](https://basovdanil.github.io/portfolio/)
