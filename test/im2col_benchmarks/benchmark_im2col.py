#!/usr/bin/env python3
"""Benchmark one-node Conv-like models with and without im2col."""

from __future__ import annotations

import argparse
import functools
import operator
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_BUILD_DIR = REPO_ROOT / "build"

TYPE_MAP = {
    1: "float",
    2: "uint8_t",
    3: "int8_t",
    4: "uint16_t",
    5: "int16_t",
    6: "int32_t",
    7: "int64_t",
    10: "_Float16",
    11: "double",
    12: "uint32_t",
    13: "uint64_t",
}

WRAPPER_C = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef INPUT_ELEMS
#error INPUT_ELEMS must be defined
#endif
#ifndef OUTPUT_ELEMS
#error OUTPUT_ELEMS must be defined
#endif
#ifndef INPUT_CTYPE
#error INPUT_CTYPE must be defined
#endif
#ifndef OUTPUT_CTYPE
#error OUTPUT_CTYPE must be defined
#endif

extern void entry(const INPUT_CTYPE* input, OUTPUT_CTYPE* output);

int main(int argc, char** argv) {
    int iterations = 1000;
    const char* output_path = NULL;
    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    if (argc > 2) {
        output_path = argv[2];
    }

    INPUT_CTYPE* input = malloc((size_t)INPUT_ELEMS * sizeof(INPUT_CTYPE));
    OUTPUT_CTYPE* output = malloc((size_t)OUTPUT_ELEMS * sizeof(OUTPUT_CTYPE));
    if (!input || !output) {
        fprintf(stderr, "allocation failed\n");
        free(input);
        free(output);
        return 1;
    }

    for (int i = 0; i < INPUT_ELEMS; i++) {
        input[i] = (INPUT_CTYPE)((i % 23) + 1);
    }

    for (int i = 0; i < 5; i++) {
        entry(input, output);
    }

    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        entry(input, output);
    }
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_ms = (elapsed * 1000.0) / iterations;

    printf("Iterations: %d\n", iterations);
    printf("Total time: %.6f seconds\n", elapsed);
    printf("Average per iteration: %.6f ms\n", avg_ms);
    printf("Throughput: %.2f iterations/sec\n", iterations / elapsed);

    if (output_path != NULL) {
        FILE* fp = fopen(output_path, "wb");
        if (!fp) {
            fprintf(stderr, "failed to open output dump file\n");
            free(input);
            free(output);
            return 1;
        }
        size_t written = fwrite(output, sizeof(OUTPUT_CTYPE), (size_t)OUTPUT_ELEMS, fp);
        fclose(fp);
        if (written != (size_t)OUTPUT_ELEMS) {
            fprintf(stderr, "failed to write output dump file\n");
            free(input);
            free(output);
            return 1;
        }
    }

    free(input);
    free(output);
    return 0;
}
"""


@dataclass(frozen=True)
class ModelMeta:
    input_ctype: str
    input_elems: int
    output_ctype: str
    output_elems: int
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    kernel_shape: tuple[int, ...]
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    pads: tuple[int, ...]
    group: int
    op_type: str


@dataclass(frozen=True)
class BenchmarkResult:
    model: str
    baseline_ms: float
    im2col_ms: float
    meta: ModelMeta
    im2col_applied: bool
    forced_im2col_ms: float | None = None
    forced_im2col_applied: bool = False

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.im2col_ms

    @property
    def forced_speedup(self) -> float | None:
        if self.forced_im2col_ms is None:
            return None
        return self.baseline_ms / self.forced_im2col_ms


@dataclass(frozen=True)
class Im2ColPhaseBreakdown:
    input_reshape_ratio: float
    core_gemm_ratio: float
    output_reshape_ratio: float

    @property
    def input_reshape_pct(self) -> float:
        return self.input_reshape_ratio * 100.0

    @property
    def core_gemm_pct(self) -> float:
        return self.core_gemm_ratio * 100.0

    @property
    def output_reshape_pct(self) -> float:
        return self.output_reshape_ratio * 100.0


def run(cmd: list[str], *, stdout_path: Path | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    stdout = subprocess.PIPE
    handle = None
    if stdout_path is not None:
        handle = stdout_path.open("w")
        stdout = handle
    try:
        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=stdout,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        print(f"command failed: {' '.join(cmd)}", file=sys.stderr)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
    finally:
        if handle is not None:
            handle.close()


def elem_count(value: onnx.ValueInfoProto) -> int:
    dims = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
    return functools.reduce(operator.mul, dims, 1)


def shape_tuple(value: onnx.ValueInfoProto) -> tuple[int, ...]:
    return tuple(dim.dim_value for dim in value.type.tensor_type.shape.dim)


def ctype(value: onnx.ValueInfoProto) -> str:
    elem_type = value.type.tensor_type.elem_type
    if elem_type not in TYPE_MAP:
        raise ValueError(f"unsupported tensor type {elem_type}")
    return TYPE_MAP[elem_type]


def model_meta(model_path: Path) -> ModelMeta:
    model = onnx.load(model_path)
    if len(model.graph.node) != 1:
        raise ValueError(f"{model_path.name}: expected exactly one graph node")

    node = model.graph.node[0]
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
    initializers = {tensor.name for tensor in model.graph.initializer}
    initializers_by_name = {tensor.name: tensor for tensor in model.graph.initializer}
    graph_inputs = [value for value in model.graph.input if value.name not in initializers]
    if len(graph_inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"{model_path.name}: expected one non-initializer input and one output")

    weight_name = node.input[1] if len(node.input) > 1 else None
    weight_shape = ()
    if weight_name and weight_name in initializers_by_name:
        weight_shape = tuple(initializers_by_name[weight_name].dims)

    kernel_shape = tuple(attrs.get("kernel_shape", weight_shape[2:4] if len(weight_shape) >= 4 else ()))
    strides = tuple(attrs.get("strides", [1, 1]))
    dilations = tuple(attrs.get("dilations", [1, 1]))
    pads = tuple(attrs.get("pads", [0, 0, 0, 0]))
    group = int(attrs.get("group", 1))

    return ModelMeta(
        input_ctype=ctype(graph_inputs[0]),
        input_elems=elem_count(graph_inputs[0]),
        output_ctype=ctype(model.graph.output[0]),
        output_elems=elem_count(model.graph.output[0]),
        input_shape=shape_tuple(graph_inputs[0]),
        output_shape=shape_tuple(model.graph.output[0]),
        kernel_shape=kernel_shape,
        strides=strides,
        dilations=dilations,
        pads=pads,
        group=group,
        op_type=node.op_type,
    )


def ensure_models() -> None:
    # Always regenerate the synthetic benchmark suite so the ONNX files stay
    # aligned with generate_models.py as cases are added or adjusted.
    run([sys.executable, str(SCRIPT_DIR / "generate_models.py")])


def write_wrapper(out_dir: Path) -> Path:
    wrapper = out_dir / "benchmark_wrapper.c"
    wrapper.write_text(WRAPPER_C)
    return wrapper


def generate_c(onnx2c: Path, model: Path, output: Path, *, im2col_mode: str | None) -> None:
    cmd = [str(onnx2c), "-l", "0"]
    if im2col_mode == "heuristic":
        cmd.extend(["-p", "im2col"])
    elif im2col_mode == "all":
        cmd.extend(["-p", "im2col_all"])
    cmd.append(str(model))
    run(cmd, stdout_path=output)


def generated_c_uses_im2col(source: Path) -> bool:
    return "Fused Im2Col + MatMul" in source.read_text()


def compile_binary(gcc: str, source: Path, wrapper: Path, binary: Path, meta: ModelMeta) -> None:
    cmd = [
        gcc,
        "-O3",
        f"-DINPUT_CTYPE={meta.input_ctype}",
        f"-DOUTPUT_CTYPE={meta.output_ctype}",
        f"-DINPUT_ELEMS={meta.input_elems}",
        f"-DOUTPUT_ELEMS={meta.output_elems}",
        "-o",
        str(binary),
        str(source),
        str(wrapper),
        "-lm",
    ]
    run(cmd)


def parse_avg_ms(output: str) -> float:
    match = re.search(r"Average per iteration:\s+([0-9.]+)\s+ms", output)
    if not match:
        raise ValueError(f"could not parse benchmark output:\n{output}")
    return float(match.group(1))


def run_binary(binary: Path, iterations: int, output_dump: Path | None = None) -> float:
    cmd = [str(binary), str(iterations)]
    if output_dump is not None:
        cmd.append(str(output_dump))
    result = run(cmd)
    return parse_avg_ms(result.stdout)


def numpy_dtype(c_type: str) -> np.dtype:
    mapping = {
        "float": np.float32,
        "double": np.float64,
        "_Float16": np.float16,
        "uint8_t": np.uint8,
        "int8_t": np.int8,
        "uint16_t": np.uint16,
        "int16_t": np.int16,
        "int32_t": np.int32,
        "int64_t": np.int64,
        "uint32_t": np.uint32,
        "uint64_t": np.uint64,
    }
    if c_type not in mapping:
        raise ValueError(f"unsupported C tensor type for numpy conversion: {c_type}")
    return np.dtype(mapping[c_type])


def compare_output_tensors(meta: ModelMeta, baseline_dump: Path, im2col_dump: Path) -> tuple[bool, str]:
    dtype = numpy_dtype(meta.output_ctype)
    baseline = np.fromfile(baseline_dump, dtype=dtype, count=meta.output_elems)
    im2col = np.fromfile(im2col_dump, dtype=dtype, count=meta.output_elems)

    if baseline.size != meta.output_elems or im2col.size != meta.output_elems:
        return False, "unexpected output dump size"

    # Always compare numerically with a fixed strict tolerance.
    atol, rtol = 1e-8, 1e-8
    baseline_cmp = baseline.astype(np.float64, copy=False)
    im2col_cmp = im2col.astype(np.float64, copy=False)
    equal = np.allclose(baseline_cmp, im2col_cmp, atol=atol, rtol=rtol, equal_nan=True)
    if equal:
        return True, ""

    # float32 paths can differ by a few ULPs because im2col uses a local accumulator
    # while baseline Conv updates y[...] directly in the inner loop.
    if meta.output_ctype == "float":
        baseline_i = baseline.view(np.int32).astype(np.int64, copy=False)
        im2col_i = im2col.view(np.int32).astype(np.int64, copy=False)
        baseline_ord = np.where(baseline_i < 0, np.int64(0x80000000) - baseline_i, baseline_i)
        im2col_ord = np.where(im2col_i < 0, np.int64(0x80000000) - im2col_i, im2col_i)
        ulp_diff = np.abs(baseline_ord - im2col_ord)
        max_ulp = int(np.max(ulp_diff))
        if max_ulp <= 2:
            return True, ""

    abs_diff = np.abs(baseline_cmp - im2col_cmp)
    max_idx = int(np.nanargmax(abs_diff))
    max_abs = float(abs_diff[max_idx])
    base_val = float(baseline_cmp[max_idx])
    im2col_val = float(im2col_cmp[max_idx])
    rel = max_abs / max(abs(base_val), 1e-30)
    return (
        False,
        (
            f"max abs diff {max_abs:.6e} (rel {rel:.6e}) at output index {max_idx}: "
            f"baseline={base_val:.6e}, im2col={im2col_val:.6e}, tol(atol={atol}, rtol={rtol})"
        ),
    )


def benchmark_model(args: argparse.Namespace, model: Path, wrapper: Path) -> BenchmarkResult:
    name = model.stem
    meta = model_meta(model)

    baseline_c = args.out_dir / f"{name}_baseline.c"
    im2col_c = args.out_dir / f"{name}_im2col.c"
    forced_c = args.out_dir / f"{name}_im2col_all.c"
    baseline_bin = args.out_dir / f"{name}_baseline"
    im2col_bin = args.out_dir / f"{name}_im2col"
    forced_bin = args.out_dir / f"{name}_im2col_all"
    baseline_dump = args.out_dir / f"{name}_baseline.out.bin"
    im2col_dump = args.out_dir / f"{name}_im2col.out.bin"
    forced_dump = args.out_dir / f"{name}_im2col_all.out.bin"

    generate_c(args.onnx2c, model, baseline_c, im2col_mode=None)
    generate_c(args.onnx2c, model, im2col_c, im2col_mode=args.im2col_mode)
    im2col_applied = generated_c_uses_im2col(im2col_c)
    compile_binary(args.gcc, baseline_c, wrapper, baseline_bin, meta)
    compile_binary(args.gcc, im2col_c, wrapper, im2col_bin, meta)

    baseline_ms = run_binary(baseline_bin, args.iterations, baseline_dump)
    im2col_ms = run_binary(im2col_bin, args.iterations, im2col_dump)

    tensors_equal, reason = compare_output_tensors(meta, baseline_dump, im2col_dump)
    if not tensors_equal:
        raise ValueError(f"{name}: output mismatch between baseline and im2col; {reason}")

    forced_im2col_ms = None
    forced_im2col_applied = False
    if args.im2col_mode == "heuristic" and not im2col_applied:
        generate_c(args.onnx2c, model, forced_c, im2col_mode="all")
        forced_im2col_applied = generated_c_uses_im2col(forced_c)
        compile_binary(args.gcc, forced_c, wrapper, forced_bin, meta)
        forced_im2col_ms = run_binary(forced_bin, args.iterations, forced_dump)

        tensors_equal, reason = compare_output_tensors(meta, baseline_dump, forced_dump)
        if not tensors_equal:
            raise ValueError(f"{name}: output mismatch between baseline and im2col_all; {reason}")

    return BenchmarkResult(
        name,
        baseline_ms,
        im2col_ms,
        meta,
        im2col_applied,
        forced_im2col_ms,
        forced_im2col_applied,
    )


def fmt_dims(dims: tuple[int, ...]) -> str:
    if not dims:
        return "-"
    return "x".join(str(dim) for dim in dims)


def estimate_im2col_phase_breakdown(meta: ModelMeta) -> Im2ColPhaseBreakdown:
    # This benchmark's im2col kernel is generated as fused loops rather than
    # explicit temporary matrices. We still report a three-stage view by
    # attributing work to: (1) input reshape/indexing, (2) MAC core, (3) output writeback.
    if len(meta.input_shape) != 4 or len(meta.output_shape) != 4 or len(meta.kernel_shape) != 2:
        return Im2ColPhaseBreakdown(0.0, 1.0, 0.0)

    batch, in_ch, _, _ = meta.input_shape
    _, out_ch, out_h, out_w = meta.output_shape
    kernel_h, kernel_w = meta.kernel_shape

    group = max(meta.group, 1)
    in_ch_per_group = in_ch // group
    kernel_elems_per_output = in_ch_per_group * kernel_h * kernel_w
    output_positions = batch * out_h * out_w
    output_elems = output_positions * out_ch

    # Work model for explicit im2col + GEMM style decomposition:
    # - input reshape: gather + index math + bounds handling
    # - core gemm: dot-product MACs
    # - output reshape: quantization/finalize + writeback
    macs = output_elems * kernel_elems_per_output
    reshape_points = output_positions * kernel_elems_per_output

    has_padding = any(p != 0 for p in meta.pads)
    has_dilation = any(d != 1 for d in meta.dilations)
    has_stride = any(s != 1 for s in meta.strides)

    reshape_unit_cost = 1.0
    if has_padding:
        reshape_unit_cost += 0.4
    if has_dilation:
        reshape_unit_cost += 0.3
    if has_stride:
        reshape_unit_cost += 0.2

    if meta.op_type == "QLinearConv":
        gemm_unit_cost = 1.0
        out_unit_cost = 4.0
    else:
        gemm_unit_cost = 1.0
        out_unit_cost = 1.0

    reshape_work = reshape_points * reshape_unit_cost
    gemm_work = macs * gemm_unit_cost
    out_work = output_elems * out_unit_cost
    total = reshape_work + gemm_work + out_work

    if total <= 0:
        return Im2ColPhaseBreakdown(0.0, 1.0, 0.0)

    return Im2ColPhaseBreakdown(
        input_reshape_ratio=reshape_work / total,
        core_gemm_ratio=gemm_work / total,
        output_reshape_ratio=out_work / total,
    )


def print_table(results: list[BenchmarkResult]) -> None:
    def fmt_optional_ms(value: float | None) -> str:
        return "-" if value is None else f"{value:.6f}"

    def fmt_optional_speedup(value: float | None) -> str:
        return "-" if value is None else f"{value:.2f}x"

    rows = [
        (
            "model",
            "op",
            "input",
            "output",
            "kernel",
            "stride",
            "dilation",
            "pads",
            "group",
            "baseline ms",
            "heur ms",
            "heur spd",
            "heur applied",
            "all ms",
            "all spd",
            "all applied",
            "im2col in%",
            "im2col gemm%",
            "im2col out%",
        ),
        (
            "-----",
            "--",
            "-----",
            "------",
            "------",
            "------",
            "--------",
            "----",
            "-----",
            "-----------",
            "-------",
            "--------",
            "------------",
            "------",
            "-------",
            "-----------",
            "----------",
            "------------",
            "----------",
        ),
    ]
    for result in results:
        phase = estimate_im2col_phase_breakdown(result.meta)
        rows.append(
            (
                result.model,
                result.meta.op_type,
                fmt_dims(result.meta.input_shape),
                fmt_dims(result.meta.output_shape),
                fmt_dims(result.meta.kernel_shape),
                fmt_dims(result.meta.strides),
                fmt_dims(result.meta.dilations),
                fmt_dims(result.meta.pads),
                str(result.meta.group),
                f"{result.baseline_ms:.6f}",
                f"{result.im2col_ms:.6f}",
                f"{result.speedup:.2f}x",
                "yes" if result.im2col_applied else "no",
                fmt_optional_ms(result.forced_im2col_ms),
                fmt_optional_speedup(result.forced_speedup),
                "yes" if result.forced_im2col_applied else "-",
                f"{phase.input_reshape_pct:.1f}%",
                f"{phase.core_gemm_pct:.1f}%",
                f"{phase.output_reshape_pct:.1f}%",
            )
        )

    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for idx, row in enumerate(rows):
        print(
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:<{widths[1]}}  "
            f"{row[2]:>{widths[2]}}  "
            f"{row[3]:>{widths[3]}}  "
            f"{row[4]:>{widths[4]}}  "
            f"{row[5]:>{widths[5]}}  "
            f"{row[6]:>{widths[6]}}  "
            f"{row[7]:>{widths[7]}}  "
            f"{row[8]:>{widths[8]}}  "
            f"{row[9]:>{widths[9]}}  "
            f"{row[10]:>{widths[10]}}  "
            f"{row[11]:>{widths[11]}}  "
            f"{row[12]:>{widths[12]}}  "
            f"{row[13]:>{widths[13]}}  "
            f"{row[14]:>{widths[14]}}  "
            f"{row[15]:>{widths[15]}}  "
            f"{row[16]:>{widths[16]}}  "
            f"{row[17]:>{widths[17]}}  "
            f"{row[18]:>{widths[18]}}"
        )
        if idx == 1:
            continue


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", "-n", type=int, default=1000, help="Measured entry() calls per binary.")
    parser.add_argument("--build-dir", type=Path, default=Path(os.environ.get("BUILD_DIR", DEFAULT_BUILD_DIR)))
    parser.add_argument("--onnx2c", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--gcc", default=os.environ.get("CC", "gcc"))
    parser.add_argument("--im2col-mode", choices=["heuristic", "all"], default="heuristic")
    parser.add_argument("--model", action="append", default=[], help="Run only models whose filename stem contains this string.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.onnx2c = args.onnx2c or args.build_dir / "onnx2c"
    args.out_dir = args.out_dir or args.build_dir / "im2col_benchmarks"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.onnx2c.exists():
        raise SystemExit(f"onnx2c not found: {args.onnx2c}")

    ensure_models()
    models = sorted(SCRIPT_DIR.glob("*.onnx"))
    if args.model:
        models = [model for model in models if any(pattern in model.stem for pattern in args.model)]
    if not models:
        raise SystemExit("no benchmark models selected")

    wrapper = write_wrapper(args.out_dir)
    results = [
        benchmark_model(args, model, wrapper)
        for model in tqdm(models, desc="Benchmarking models", unit="model")
    ]
    print_table(results)


if __name__ == "__main__":
    main()
