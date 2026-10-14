#!/usr/bin/env python3
"""Benchmark the onnx2c tensor-memory strategies on a single ONNX model.

Runs onnx2c once per activation-memory optimization (none, unionize, tensor arena)
on the same model, captures stdout/stderr, parses the planner-reported
memory metrics and prints one compact table. Machine-readable output can be
written with ``--csv``.

The intermediate-memory numbers come from each run's "Tensor memory allocation"
metrics block; the script does not re-derive allocations.

Each mode reports the bytes allocated by the selected strategy.

Generated C code is kept in a temporary directory that is removed when the
script finishes, unless an explicit output directory is requested.
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Modes are run (and reported) in this order.
MODES = ("none", "union", "arena")
ARENA_STRATEGIES = ("first-fit", "memory-schedule")

# Log block emitted by toC::Graph::log_tensor_arena_metrics(), and the metric
# labels -> result keys it contains.
PLANNER_BLOCK_HEADER = "Tensor memory allocation:"
PLANNER_METRICS = {
    "eligible tensors": "eligible_tensor_count",
    "total intermediate bytes": "total_intermediate_bytes",
    "peak-live lower bound": "peak_live_lower_bound",
    "allocated bytes": "allocated_bytes",
}

# Columns for the --csv output. Percentages are kept at full precision.
CSV_COLUMNS = (
    "model",
    "onnx2c",
    "eligible_tensor_count",
    "total_intermediate_bytes",
    "peak_live_lower_bound",
    "none_bytes",
    "union_bytes",
    "arena_bytes",
    "arena_vs_union_bytes",
    "arena_vs_union_pct",
    "arena_gap_to_lower_bound_bytes",
    "arena_gap_to_lower_bound_pct",
)


def parse_planner_metrics(text):
    """Extract the 'Tensor memory allocation:' metrics block from log text.

    Returns a dict mapping metric result keys (see PLANNER_METRICS) to ints.
    Returns an empty dict when no planner block is present. Parsing is robust
    against unrelated log output: each known metric is searched for inside the
    lines following the block header until all are found or input ends.
    """
    metrics = {}
    line_after_header = False
    for line in text.splitlines():
        if PLANNER_BLOCK_HEADER in line:
            line_after_header = True
            continue
        if not line_after_header:
            continue
        for label, key in PLANNER_METRICS.items():
            if key in metrics:
                continue
            match = re.search(re.escape(label) + r"\s*:\s*(\d+)", line)
            if match:
                metrics[key] = int(match.group(1))
        if len(metrics) == len(PLANNER_METRICS):
            break
    return metrics


def derive_metrics(planner, allocations):
    """Combine the planner metrics into the per-mode and comparison numbers."""
    peak = planner.get("peak_live_lower_bound", 0)
    union = allocations["union"]
    arena = allocations["arena"]

    arena_vs_union = arena - union
    arena_vs_union_pct = 100.0 * arena_vs_union / union if union else 0.0
    gap = arena - peak
    gap_pct = 100.0 * gap / peak if peak else 0.0

    return {
        "none_bytes": allocations["none"],
        "union_bytes": union,
        "arena_bytes": arena,
        "arena_vs_union_bytes": arena_vs_union,
        "arena_vs_union_pct": arena_vs_union_pct,
        "arena_gap_to_lower_bound_bytes": gap,
        "arena_gap_to_lower_bound_pct": gap_pct,
    }


def format_bytes(value):
    """Format a byte count for the terminal, e.g. '-9,766,912 B'."""
    sign = "-" if value < 0 else ""
    return f"{sign}{abs(value):,} B"


def format_percent(value):
    return f"{value:.2f}%"


def format_report(model_name, planner, derived):
    """Build the human-readable report text."""
    mode_rows = [
        ("none", derived["none_bytes"]),
        ("union", derived["union_bytes"]),
        ("arena", derived["arena_bytes"]),
    ]

    lines = [
        f"Model: {model_name}",
        "",
        f"{'Mode':<10}{'Memory'}",
    ]
    for mode, size in mode_rows:
        lines.append(f"{mode:<10}{format_bytes(size)}")

    lines.append("")
    summary = [
        ("Eligible tensors:", f"{planner.get('eligible_tensor_count', 0):,}"),
        ("Total intermediate bytes:", format_bytes(planner.get("total_intermediate_bytes", 0))),
        ("Peak-live lower bound:", format_bytes(planner.get("peak_live_lower_bound", 0))),
        ("Arena vs union:", f"{format_bytes(derived['arena_vs_union_bytes'])} ({format_percent(derived['arena_vs_union_pct'])})"),
        ("Arena gap to lower bound:", f"{format_bytes(derived['arena_gap_to_lower_bound_bytes'])} ({format_percent(derived['arena_gap_to_lower_bound_pct'])})"),
    ]
    label_width = max(len(label) for label, _ in summary) + 1
    for label, value in summary:
        lines.append(f"{label:<{label_width}}{value}")
    return "\n".join(lines)


def find_onnx2c(explicit=None):
    """Locate the onnx2c binary from --onnx2c, $ONNX2C or the build dirs."""
    if explicit:
        return Path(explicit).resolve()
    from_env = os.environ.get("ONNX2C")
    if from_env:
        return Path(from_env).resolve()
    repo_root = Path(__file__).resolve().parent.parent
    for relative in ("build/onnx2c", "build-arena/onnx2c", "onnx2c"):
        candidate = repo_root / relative
        if candidate.is_file():
            return candidate
    return None


def run_mode(onnx2c, model, mode, workdir, arena_strategy=None):
    """Run onnx2c once for one tensor-memory mode.

    Returns the combined stdout+stderr text of the run. Raises a
    RuntimeError with a clear message if generation fails.
    """
    suffix = f"_{arena_strategy}" if arena_strategy else ""
    c_file = workdir / f"{model.stem}_{mode}{suffix}.c"
    log_file = workdir / f"{model.stem}_{mode}{suffix}.log"
    optimizations = {
        "none": "fold_casts",
        "union": "fold_casts,unionize",
        "arena": "fold_casts,tensor_arena",
    }[mode]
    if arena_strategy == "memory-schedule":
        optimizations += ",memory_schedule"
    command = [str(onnx2c), "-l2", f"--optimizations={optimizations}"]
    command.append(str(model))
    try:
        with open(c_file, "w") as c_stream, open(log_file, "w") as log_stream:
            result = subprocess.run(command, stdout=c_stream, stderr=log_stream)
    except OSError as exc:
        raise RuntimeError(f"failed to run onnx2c in {mode} mode: {exc}")
    if result.returncode != 0:
        tail = ""
        try:
            tail = log_file.read_text().splitlines()[-10:]
        except OSError:
            pass
        raise RuntimeError(
            f"onnx2c {mode} mode exited with code {result.returncode}\n"
            f"stderr tail:\n" + "\n".join(tail)
        )
    return c_file.read_text() + log_file.read_text()


def write_csv(path, row):
    stream = sys.stdout if path == "-" else open(path, "w", newline="")
    try:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow({column: row.get(column, 0) for column in CSV_COLUMNS})
    finally:
        if stream is not sys.stdout:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="benchmark_tensor_memory.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="path to the ONNX model")
    parser.add_argument(
        "--onnx2c",
        help="path to the onnx2c binary (default: $ONNX2C, then build dirs)",
    )
    parser.add_argument(
        "--outdir", "--output-dir", dest="outdir",
        help="keep generated artifacts in DIR instead of a temporary directory",
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        help="write machine-readable results as CSV to FILE ('-' for stdout)",
    )
    parser.add_argument(
        "--arena-strategies",
        help="also benchmark first-fit or memory-schedule (comma-separated)",
    )
    args = parser.parse_args(argv)

    model = Path(args.model)
    if not model.is_file():
        parser.error(f"model file not found: {model}")

    onnx2c = find_onnx2c(args.onnx2c)
    if onnx2c is None or not onnx2c.is_file():
        parser.error(
            "onnx2c binary not found; pass --onnx2c or set the ONNX2C "
            "environment variable"
        )

    tempdir = None
    workdir = Path(args.outdir) if args.outdir else None
    if workdir is None:
        tempdir = Path(tempfile.mkdtemp(prefix="onnx2c_benchmark_"))
        workdir = tempdir
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    try:
        metrics_by_mode = {}
        for mode in MODES:
            output = run_mode(onnx2c, model, mode, workdir)
            metrics_by_mode[mode] = parse_planner_metrics(output)

        incomplete = [mode for mode, metrics in metrics_by_mode.items()
                      if len(metrics) != len(PLANNER_METRICS)]
        if incomplete:
            raise RuntimeError("incomplete tensor-memory metrics for: " + ", ".join(incomplete))
        planner = metrics_by_mode["arena"]
        allocations = {mode: metrics["allocated_bytes"] for mode, metrics in metrics_by_mode.items()}
        derived = derive_metrics(planner, allocations)

        print(format_report(model.name, planner, derived))
        if args.arena_strategies:
            strategies = [x.strip() for x in args.arena_strategies.split(",") if x.strip()]
            invalid = [x for x in strategies if x not in ARENA_STRATEGIES]
            if invalid:
                raise RuntimeError("unknown arena strategy: " + ", ".join(invalid))
            print("\nStrategy             Peak live       Arena       vs first-fit")
            print("-" * 67)
            baseline = derived["arena_bytes"]
            for strategy in strategies:
                output = run_mode(onnx2c, model, "arena", workdir, strategy)
                metrics = parse_planner_metrics(output)
                if len(metrics) != len(PLANNER_METRICS):
                    raise RuntimeError(f"incomplete metrics for arena strategy {strategy}")
                delta = metrics["allocated_bytes"] - baseline
                print(f"{strategy:<21}{format_bytes(metrics['peak_live_lower_bound']):<16}{format_bytes(metrics['allocated_bytes']):<12}{format_bytes(delta)}")

        if args.csv:
            row = {"model": model.name, "onnx2c": str(onnx2c)}
            row.update(planner)
            row.update(derived)
            write_csv(args.csv, row)
    except RuntimeError as exc:
        print(f"benchmark_tensor_memory: error: {exc}", file=sys.stderr)
        return 1
    finally:
        if tempdir is not None:
            shutil.rmtree(tempdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
