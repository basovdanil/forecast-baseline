import csv
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forecast


class ModelTest(unittest.TestCase):
    def test_naive_repeats_last_value(self):
        self.assertEqual(forecast.forecast("naive", [1, 2, 3], 3), [3, 3, 3])

    def test_mean_is_average_of_history(self):
        self.assertEqual(forecast.forecast("mean", [2, 4, 6], 2), [4, 4])

    def test_ma7_uses_last_seven_points(self):
        history = [1, 1, 1, 1, 1, 1, 1, 8]
        self.assertEqual(forecast.forecast("ma7", history, 2), [2, 2])

    def test_seasonal_week_cycles_last_week(self):
        history = [10, 11, 12, 13, 14, 15, 16]
        self.assertEqual(forecast.forecast("seasonal7", history, 9),
                         [10, 11, 12, 13, 14, 15, 16, 10, 11])

    def test_trend_extrapolates_a_line(self):
        predicted = forecast.forecast("trend", [1, 2, 3, 4, 5], 3)
        self.assertEqual([round(p, 6) for p in predicted], [6, 7, 8])

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            forecast.forecast("arima", [1, 2], 1)


class ErrorTest(unittest.TestCase):
    def test_mae_and_mape(self):
        mae, mape = forecast.errors([10, 20], [12, 20])
        self.assertEqual(mae, 1.0)
        self.assertAlmostEqual(mape, 10.0)

    def test_zero_actuals_are_skipped_for_mape(self):
        mae, mape = forecast.errors([0, 10], [2, 12])
        self.assertEqual(mae, 2.0)
        self.assertAlmostEqual(mape, 20.0)

    def test_all_zero_actuals_return_none(self):
        self.assertIsNone(forecast.errors([0, 0], [1, 1]))


class BacktestTest(unittest.TestCase):
    def test_perfect_line_prefers_trend(self):
        report = forecast.backtest(list(range(1, 21)), horizon=1, min_train=7)
        self.assertEqual(report["best"], "trend")
        self.assertEqual(report["origins"], 13)
        self.assertEqual(report["methods"]["trend"]["mae"], 0.0)
        self.assertEqual(report["methods"]["trend"]["origins"], 13)

    def test_short_series_raises(self):
        with self.assertRaises(ValueError):
            forecast.backtest(list(range(10)), horizon=7, min_train=7)


class SeriesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_series_normalizes_numbers(self):
        path = os.path.join(self.dir, "s.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            f.write('date,value\n2026-01-01,"1 234,5"\n2026-01-02,7\n')
        series = forecast.read_series(path)
        self.assertEqual(series, [("2026-01-01", 1234.5), ("2026-01-02", 7.0)])

    def test_read_series_without_value_column(self):
        path = os.path.join(self.dir, "s.csv")
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write("date,amount\n2026-01-01,10\n")
        self.assertIsNone(forecast.read_series(path))

    def test_cli_missing_file_returns_3(self):
        code = forecast.main(["backtest", "--input", os.path.join(self.dir, "нет.csv")])
        self.assertEqual(code, 3)

    def test_cli_creates_json_and_csv(self):
        path = os.path.join(self.dir, "s.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "value"])
            for i in range(40):
                writer.writerow([f"2026-01-{i + 1:02d}", 100 + i])
        out = os.path.join(self.dir, "metrics.json")
        table = os.path.join(self.dir, "metrics.csv")
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = forecast.main(["backtest", "--input", path, "--horizon", "7",
                                  "--min-train", "28", "--out", out, "--csv", table,
                                  "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["origins"], 6)
        self.assertEqual(payload["best"], "trend")
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["rows"], 40)
        with open(table, encoding="utf-8-sig", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 5)

    def test_cli_short_series_returns_4(self):
        path = os.path.join(self.dir, "s.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "value"])
            for i in range(10):
                writer.writerow([f"2026-01-{i + 1:02d}", 100 + i])
        self.assertEqual(forecast.main(["backtest", "--input", path]), 4)


if __name__ == "__main__":
    unittest.main()
