#!/usr/bin/env python3
"""Create CSV and summary tables for im2col policy benchmark JSON."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path


def runtime_ms(row: dict) -> float:
    unit = row.get("time_unit", "ns")
    value = float(row.get("real_time", row.get("cpu_time", 0.0)))
    scale = {
        "ns": 1e-6,
        "us": 1e-3,
        "ms": 1.0,
        "s": 1000.0,
    }.get(unit)
    if scale is None:
        raise ValueError(f"unsupported benchmark time_unit: {unit}")
    return value * scale


def object_size(model: str, policy: str, benchmark_dir: Path) -> tuple[str, str, str]:
    obj = benchmark_dir.parent / "CMakeFiles" / "dummy.dir" / f"{model}_{policy}.c.o"
    if not obj.exists():
        return "", "", ""

    try:
        result = subprocess.run(
            ["/usr/bin/size", str(obj)],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        return "", "", ""

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return "", "", ""

    header = re.split(r"\s+", lines[0])
    values = re.split(r"\s+", lines[1])

    if header[:3] == ["text", "data", "bss"] and len(values) >= 3:
        return values[0], values[1], values[2]

    # macOS object output: __TEXT __DATA __OBJC others dec hex.
    if "__TEXT" in header and "__DATA" in header:
        text = values[header.index("__TEXT")]
        data = values[header.index("__DATA")]
        return text, data, ""

    return "", "", ""


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: summarize_im2col_policies.py input.json output.csv", file=sys.stderr)
        return 2

    input_json = Path(sys.argv[1])
    output_csv = Path(sys.argv[2])
    benchmark_dir = input_json.resolve().parent
    data = json.loads(input_json.read_text())

    rows = []
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "")
        if "/" not in name:
            continue
        model, policy = name.split("/", 1)
        if policy not in {"none", "all", "heuristic"}:
            continue
        text_bytes, data_bytes, bss_bytes = object_size(model, policy, benchmark_dir)
        peak_memory = int(bench.get("peak_memory_bytes_measured", 0))
        rows.append(
            {
                "model": model,
                "policy": policy,
                "runtime_ms": f"{runtime_ms(bench):.6f}",
                "im2col_extra_bytes": int(bench.get("im2col_extra_bytes_theoretical", 0)),
                "peak_memory_bytes": peak_memory if peak_memory > 0 else "",
                "code_text_bytes": text_bytes,
                "data_bytes": data_bytes,
                "bss_bytes": bss_bytes,
                "num_convs": int(bench.get("num_convs", 0)),
                "num_im2col_convs": int(bench.get("num_im2col_convs", 0)),
                "im2col_fraction": f"{float(bench.get('im2col_fraction', 0.0)):.6f}",
            }
        )

    rows.sort(key=lambda r: (r["model"], {"none": 0, "all": 1, "heuristic": 2}[r["policy"]]))
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "policy",
                "runtime_ms",
                "im2col_extra_bytes",
                "peak_memory_bytes",
                "code_text_bytes",
                "data_bytes",
                "bss_bytes",
                "num_convs",
                "num_im2col_convs",
                "im2col_fraction",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    by_model: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["policy"]] = row

    print("")
    print("Im2Col policy summary")
    print("model,none_ms,all_ms,heuristic_ms,none_bytes,all_bytes,heuristic_bytes,all_runtime_vs_none,heuristic_runtime_vs_none,all_memory_overhead_vs_none,heuristic_memory_overhead_vs_none")
    for model in sorted(by_model):
        policies = by_model[model]
        if not {"none", "all", "heuristic"} <= set(policies):
            continue
        none = policies["none"]
        all_ = policies["all"]
        heuristic = policies["heuristic"]
        none_ms = float(none["runtime_ms"])
        none_bytes = int(none["im2col_extra_bytes"])

        def norm_ms(row: dict) -> float:
            return float(row["runtime_ms"]) / none_ms if none_ms else 0.0

        def norm_bytes(row: dict) -> str:
            row_bytes = int(row["im2col_extra_bytes"])
            if none_bytes == 0:
                return "inf" if row_bytes else "1.000000"
            return f"{row_bytes / none_bytes:.6f}"

        print(
            f"{model},{none['runtime_ms']},{all_['runtime_ms']},{heuristic['runtime_ms']},"
            f"{none['im2col_extra_bytes']},{all_['im2col_extra_bytes']},{heuristic['im2col_extra_bytes']},"
            f"{norm_ms(all_):.6f},{norm_ms(heuristic):.6f},"
            f"{norm_bytes(all_)},{norm_bytes(heuristic)}"
        )

    print(f"\nWrote CSV: {output_csv}")
    print("peak_memory_bytes is measured process max RSS for the combined benchmark binary.")
    print("code_text_bytes/data_bytes/bss_bytes come from /usr/bin/size on each generated policy object.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
