#!/usr/bin/env python3
"""Unit tests for benchmark_tensor_memory.py.

Run from the scripts/ directory:
    python3 -m unittest test_benchmark_tensor_memory -v
"""

import unittest
from contextlib import redirect_stdout
from io import StringIO

import benchmark_tensor_memory as bm

RESNET50_PLANNER = {
    "eligible_tensor_count": 414,
    "total_intermediate_bytes": 252680768,
    "peak_live_lower_bound": 111730592,
    "union_baseline_bytes": 121497504,
    "arena_bytes": 111730592,
}


class ParsePlannerMetricsTests(unittest.TestCase):
    def test_aixlog_verbose_format(self):
        log = (
            "2026-08-28 12-11-22.000 [Warn] (getIoTensor) Graph input dimension not specified!\n"
            "2026-08-28 12-11-22.000 [Info] (unionize_tensors) Running Unionize optimization pass\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics) Tensor arena planner:\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   eligible tensors:          105\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   total intermediate bytes:  33213568\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   peak-live lower bound:     11240864\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   union baseline bytes:      14384608\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   arena bytes:               11240864\n"
            "2026-08-28 12-11-22.000 [Info] (log_tensor_arena_metrics)   savings vs union:          21.85%\n"
            "2026-08-28 12-11-22.000 [Info] (unrelated) trailing log line\n"
        )
        metrics = bm.parse_planner_metrics(log)
        self.assertEqual(metrics["eligible_tensor_count"], 105)
        self.assertEqual(metrics["total_intermediate_bytes"], 33213568)
        self.assertEqual(metrics["peak_live_lower_bound"], 11240864)
        self.assertEqual(metrics["union_baseline_bytes"], 14384608)
        self.assertEqual(metrics["arena_bytes"], 11240864)

    def test_terse_aixlog_format(self):
        log = (
            "Info2026-08-28 12-02-40.887log_tensor_arena_metricsTensor arena planner:\n"
            "Info2026-08-28 12-02-40.887log_tensor_arena_metrics  eligible tensors:          2\n"
            "Info2026-08-28 12-02-40.887log_tensor_arena_metrics  total intermediate bytes:  32\n"
            "Info2026-08-28 12-02-40.887log_tensor_arena_metrics  peak-live lower bound:     32\n"
            "Info2026-08-28 12-02-40.887log_tensor_arena_metrics  union baseline bytes:      32\n"
            "Info2026-08-28 12-02-40.887log_tensor_arena_metrics  arena bytes:               32\n"
        )
        metrics = bm.parse_planner_metrics(log)
        self.assertEqual(metrics["arena_bytes"], 32)
        self.assertEqual(metrics["eligible_tensor_count"], 2)

    def test_no_block(self):
        self.assertEqual(bm.parse_planner_metrics("only unrelated log lines"), {})

    def test_empty_input(self):
        self.assertEqual(bm.parse_planner_metrics(""), {})


class DeriveMetricsTests(unittest.TestCase):
    def test_resnet50(self):
        derived = bm.derive_metrics(RESNET50_PLANNER)
        self.assertEqual(derived["none_bytes"], 252680768)
        self.assertEqual(derived["union_bytes"], 121497504)
        self.assertEqual(derived["arena_bytes"], 111730592)
        self.assertEqual(derived["arena_vs_union_bytes"], -9766912)
        self.assertAlmostEqual(derived["arena_vs_union_pct"], -8.038, places=1)
        self.assertEqual(derived["arena_gap_to_lower_bound_bytes"], 0)
        self.assertEqual(derived["arena_gap_to_lower_bound_pct"], 0.0)

    def test_zero_union_no_division_by_zero(self):
        planner = dict(RESNET50_PLANNER)
        planner["union_baseline_bytes"] = 0
        planner["peak_live_lower_bound"] = 0
        derived = bm.derive_metrics(planner)
        self.assertEqual(derived["arena_vs_union_pct"], 0.0)
        self.assertEqual(derived["arena_gap_to_lower_bound_pct"], 0.0)


class FormattingTests(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(bm.format_bytes(0), "0 B")
        self.assertEqual(bm.format_bytes(111730592), "111,730,592 B")
        self.assertEqual(bm.format_bytes(-9766912), "-9,766,912 B")

    def test_format_percent(self):
        self.assertEqual(bm.format_percent(-8.0386), "-8.04%")
        self.assertEqual(bm.format_percent(0.0), "0.00%")

    def test_format_report(self):
        derived = bm.derive_metrics(RESNET50_PLANNER)
        report = bm.format_report("resnet50.onnx", RESNET50_PLANNER, derived)
        self.assertIn("Model: resnet50.onnx", report)
        self.assertIn("none      252,680,768 B", report)
        self.assertIn("union     121,497,504 B", report)
        self.assertIn("arena     111,730,592 B", report)
        self.assertIn("Eligible tensors:         414", report)
        self.assertIn("Total intermediate bytes: 252,680,768 B", report)
        self.assertIn("Peak-live lower bound:    111,730,592 B", report)
        self.assertIn("Arena vs union:           -9,766,912 B (-8.04%)", report)
        self.assertIn("Arena gap to lower bound: 0 B (0.00%)", report)


class CsvTests(unittest.TestCase):
    def test_write_csv(self):
        derived = bm.derive_metrics(RESNET50_PLANNER)
        row = {"model": "resnet50.onnx", "onnx2c": "/bin/onnx2c"}
        row.update(RESNET50_PLANNER)
        row.update(derived)
        stream = StringIO()
        with redirect_stdout(stream):
            bm.write_csv("-", row)
        lines = stream.getvalue().strip().splitlines()
        self.assertEqual(lines[0], ",".join(bm.CSV_COLUMNS))
        self.assertIn("resnet50.onnx", lines[1])
        self.assertIn(str(RESNET50_PLANNER["arena_bytes"]), lines[1])


if __name__ == "__main__":
    unittest.main()