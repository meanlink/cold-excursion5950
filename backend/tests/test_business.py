"""冷链温区偏离调查的算法单测。

从 backend/ 目录运行：python tests/test_business.py
需要 Python 3.12（开发机 3.11 亦可）、NumPy、SciPy。
覆盖：固定输入、严格 JSON、空数据、参数边界、参数改变真实结果、已知统计案例、
批次比较模式的等权口径、整批重抽样区间、确定性与拒绝条件、旧数据兼容。
"""
import json
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from scipy import stats

from src.services import business
from src.services.business import analyze, metadata, seed_records


def make_route(route, temps, batches=None):
    rows = []
    for i, t in enumerate(temps):
        row = {"sample": f"{route}-{i+1:03}", "route": route, "minute": i * 2,
               "temperature": round(float(t), 3)}
        if batches is not None:
            row["batch"] = batches[i]
        rows.append(row)
    return rows


def batch_dataset(route_a, route_b, upper=8.0):
    """每条路线 3 批、每批 4 点，route_b 整体更高。"""
    rows = []
    for route, means in [("A", route_a), ("B", route_b)]:
        for b, m in enumerate(means):
            for k in range(4):
                rows.append({"sample": f"{route}-{b}-{k}", "route": route,
                             "minute": (b * 4 + k) * 2,
                             "temperature": float(m + (1.2 if k == 0 else 0.0)),
                             "batch": f"{route}-B{b+1}"})
    return rows


class SeedAndPointModeTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        first, second = seed_records(), seed_records()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 240)
        self.assertTrue(all("batch" in r for r in first))

    def test_metadata_has_both_modes_and_optional_batch_column(self):
        m = metadata()
        params = {p["key"]: p for p in m["parameters"]}
        self.assertEqual(params["mode"]["default"], "point")
        self.assertEqual([o["value"] for o in params["mode"]["options"]], ["point", "batch"])
        batch_col = next(c for c in m["columns"] if c["key"] == "batch")
        self.assertTrue(batch_col["optional"])
        self.assertIn("batch_method", m)

    def test_point_mode_known_ks_case(self):
        rng = random.Random(7)
        a = make_route("路线A", [rng.gauss(5, 0.5) for _ in range(40)])
        b = make_route("路线B", [rng.gauss(9, 0.5) for _ in range(40)])
        result = analyze(a + b, {"upper": 8, "alpha": 0.05})
        expected = stats.ks_2samp(np.array([r["temperature"] for r in a]),
                                  np.array([r["temperature"] for r in b]),
                                  alternative="two-sided")
        self.assertAlmostEqual(result["metrics"][0]["value"], round(float(expected.statistic), 5), places=5)
        self.assertAlmostEqual(float(result["metrics"][1]["value"]), float(f"{expected.pvalue:.6g}"), places=8)
        # 与旧口径一致的表结构仍然存在。
        self.assertEqual([c["key"] for c in result["table"]["columns"]],
                         ["route", "count", "mean", "p95", "over", "rate"])

    def test_point_mode_ignores_batch_field_and_matches_legacy_calc(self):
        rows = seed_records()
        legacy = [dict(r) for r in rows]
        for r in legacy:
            r.pop("batch")
        with_batch = analyze(rows, {"upper": 8, "alpha": 0.05, "mode": "point"})
        without_batch = analyze(legacy, {"upper": 8, "alpha": 0.05})
        self.assertEqual(with_batch, without_batch)
        default_mode = analyze(legacy, {})
        self.assertEqual(default_mode, without_batch)

    def test_results_are_strict_json(self):
        for mode in ("point", "batch"):
            result = analyze(seed_records(), {"upper": 8, "mode": mode})
            json.dumps(result, allow_nan=False)

    def test_empty_and_insufficient_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "至少 10"):
            analyze(make_route("A", [5.0] * 3) + make_route("B", [5.0] * 3), {})
        with self.assertRaisesRegex(ValueError, "两条路线"):
            analyze(make_route("A", [5.0] * 12), {})

    def test_parameter_bounds_rejected(self):
        rows = seed_records()
        with self.assertRaisesRegex(ValueError, "上界范围"):
            analyze(rows, {"upper": 100})
        with self.assertRaisesRegex(ValueError, "显著性水平"):
            analyze(rows, {"alpha": 0.5})
        with self.assertRaisesRegex(ValueError, "比较模式"):
            analyze(rows, {"mode": "mystery"})
        with self.assertRaisesRegex(ValueError, "数值"):
            analyze(rows, {"upper": "hot"})

    def test_parameter_change_changes_result(self):
        rows = seed_records()
        r1 = analyze(rows, {"upper": 8})
        r2 = analyze(rows, {"upper": 7})
        self.assertNotEqual(r1["metrics"], r2["metrics"])


class BatchModeRejectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = batch_dataset([5.0, 5.2, 5.4], [7.0, 7.2, 7.4])

    def test_missing_batch_column_rejected_with_reason(self):
        legacy = [dict(r) for r in self.rows]
        for r in legacy:
            r.pop("batch")
        with self.assertRaisesRegex(ValueError, "批号"):
            analyze(legacy, {"mode": "batch", "upper": 8})

    def test_blank_batch_value_rejected(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["batch"] = "   "
        with self.assertRaisesRegex(ValueError, "批号"):
            analyze(rows, {"mode": "batch", "upper": 8})

    def test_fewer_than_three_batches_rejected_and_names_route(self):
        rows = [r for r in self.rows if r["batch"] != "B-B3"]
        with self.assertRaisesRegex(ValueError, "至少 3 个运输批次") as ctx:
            analyze(rows, {"mode": "batch", "upper": 8})
        self.assertIn("B", str(ctx.exception))
        # 两条路线都不足时错误信息同时点名。
        rows2 = [r for r in self.rows if r["batch"] in ("A-B1", "B-B1")]
        with self.assertRaisesRegex(ValueError, "A 仅 1 批"):
            analyze(rows2, {"mode": "batch", "upper": 8})

    def test_non_finite_temperature_rejected(self):
        rows = [dict(r) for r in self.rows]
        rows[0]["temperature"] = float("nan")
        with self.assertRaisesRegex(ValueError, "有限温度"):
            analyze(rows, {"mode": "batch", "upper": 8})

    def test_wrong_route_count_rejected(self):
        rows = [r for r in self.rows if r["route"] == "A"]
        with self.assertRaisesRegex(ValueError, "两条路线"):
            analyze(rows, {"mode": "batch", "upper": 8})


class BatchModeCalculationTests(unittest.TestCase):
    def setUp(self):
        # A 批均值超温率 0%/25%/25%（每批 4 点，每批第一个点 6.2 或 6.4 ≤ 8? 构造显式数据更稳）
        rows = []
        plan = {
            "A": {"A-B1": [5, 5, 5, 9], "A-B2": [5, 5, 9, 9], "A-B3": [5, 9, 9, 9]},
            "B": {"B-B1": [9, 9, 9, 9], "B-B2": [9, 9, 9, 5], "B-B3": [9, 9, 5, 5]},
        }
        for route, batches in plan.items():
            for b, temps in batches.items():
                for k, t in enumerate(temps):
                    rows.append({"sample": f"{route}-{b}-{k}", "route": route,
                                 "minute": k * 2, "temperature": float(t), "batch": b})
        self.rows = rows

    def test_batch_rates_equal_weight_and_table_values(self):
        result = analyze(self.rows, {"mode": "batch", "upper": 8})
        rates = {(r["route"], r["batch"]): r for r in result["table"]["rows"]}
        self.assertEqual(rates[("A", "A-B1")]["rate"], 25.0)
        self.assertEqual(rates[("A", "A-B1")]["over"], 1)
        self.assertEqual(rates[("A", "A-B1")]["count"], 4)
        self.assertEqual(rates[("A", "A-B2")]["rate"], 50.0)
        self.assertEqual(rates[("A", "A-B3")]["rate"], 75.0)
        self.assertEqual(rates[("B", "B-B1")]["rate"], 100.0)
        self.assertEqual(rates[("B", "B-B2")]["rate"], 75.0)
        self.assertEqual(rates[("B", "B-B3")]["rate"], 50.0)
        # 各批等权：A=(25+50+75)/3=50，B=(100+75+50)/3=75。
        metrics = {m["label"]: m["value"] for m in result["metrics"]}
        self.assertEqual(metrics["A 批均值超温率"], 50.0)
        self.assertEqual(metrics["B 批均值超温率"], 75.0)
        # 差值按排序后的 A−B：50−75=−25。
        diff_metric = next(m for m in result["metrics"] if "均值差" in m["label"])
        self.assertTrue(diff_metric["value"].startswith("["))
        self.assertIn("-25.00", result["summary"])

    def test_confidence_intervals_cover_bootstrap_quantiles(self):
        result = analyze(self.rows, {"mode": "batch", "upper": 8})
        # 独立用相同种子重算 bootstrap，核对区间口径（2.5/97.5 百分位）。
        grouped = business._collect_batches(self.rows, 8.0)
        props = {n: np.array([grouped[n][b]["rate"] for b in sorted(grouped[n])])
                 for n in sorted(grouped)}
        rng = np.random.default_rng(business._deterministic_seed(grouped))
        boot = {}
        for n in sorted(grouped):
            idx = rng.integers(0, len(props[n]), size=(business.BOOTSTRAP_REPLICATES, len(props[n])))
            boot[n] = props[n][idx].mean(axis=1)
        diffs = boot["A"] - boot["B"]
        low, high = (round(float(x), 3) for x in np.quantile(diffs, [0.025, 0.975]))
        diff_metric = next(m for m in result["metrics"] if "均值差" in m["label"])
        self.assertEqual(diff_metric["value"], f"[{low}, {high}]")
        # 区间端点必须落在批比例支持范围内。
        self.assertGreaterEqual(low, -100.0)
        self.assertLessEqual(high, 100.0)

    def test_deterministic_across_shuffle_and_repeat(self):
        r1 = analyze(self.rows, {"mode": "batch", "upper": 8})
        r2 = analyze(self.rows, {"mode": "batch", "upper": 8})
        shuffled = self.rows[:]
        random.Random(99).shuffle(shuffled)
        r3 = analyze(shuffled, {"mode": "batch", "upper": 8})
        self.assertEqual(r1, r2)
        self.assertEqual(r1, r3)

    def test_batch_ids_scoped_within_route(self):
        # 两条路线使用相同批号名也必须分别归集，不串路线。
        rows = []
        for route, shift in [("A", -2.0), ("B", 2.0)]:
            for b in range(3):
                for k in range(4):
                    rows.append({"sample": f"{route}-{b}-{k}", "route": route, "minute": k * 2,
                                 "temperature": 8.0 + shift, "batch": f"COMMON-{b+1}"})
        result = analyze(rows, {"mode": "batch", "upper": 8})
        routes = sorted({r["route"] for r in result["table"]["rows"]})
        self.assertEqual(routes, ["A", "B"])
        self.assertEqual(len(result["table"]["rows"]), 6)
        metrics = {m["label"]: m["value"] for m in result["metrics"]}
        self.assertEqual(metrics["A 批均值超温率"], 0.0)
        self.assertEqual(metrics["B 批均值超温率"], 100.0)

    def test_seed_dataset_runs_in_both_modes_and_chart_is_bar(self):
        result = analyze(seed_records(), {"mode": "batch", "upper": 8})
        self.assertEqual(result["chart"]["type"], "bar")
        self.assertEqual(len(result["chart"]["series"]), 2)
        for s in result["chart"]["series"]:
            self.assertEqual(len(s["values"]), 10)
            self.assertTrue(all(v is not None for v in s["values"]))
        self.assertIn("bootstrap", result["method"])

    def test_upper_changes_batch_rates(self):
        r1 = analyze(self.rows, {"mode": "batch", "upper": 8})
        r2 = analyze(self.rows, {"mode": "batch", "upper": 4})
        self.assertNotEqual(
            [row["rate"] for row in r1["table"]["rows"]],
            [row["rate"] for row in r2["table"]["rows"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
