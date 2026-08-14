#!/usr/bin/env python3
"""Generate deterministic Conv benchmark ONNX models and test data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ConvBenchmark:
    name: str
    x_shape: tuple[int, int, int, int]
    w_shape: tuple[int, int, int, int]
    y_shape: tuple[int, int, int, int]
    pads: tuple[int, int, int, int]
    strides: tuple[int, int]
    groups: int = 1


CASES = [
    ConvBenchmark(
        "benchmark_conv_resnet_stem_7x7",
        (1, 3, 224, 224),
        (64, 3, 7, 7),
        (1, 64, 112, 112),
        (3, 3, 3, 3),
        (2, 2),
    ),
    ConvBenchmark(
        "benchmark_conv_resnet_3x3",
        (1, 64, 56, 56),
        (64, 64, 3, 3),
        (1, 64, 56, 56),
        (1, 1, 1, 1),
        (1, 1),
    ),
    ConvBenchmark(
        "benchmark_conv_resnet_bottleneck_1x1",
        (1, 256, 28, 28),
        (64, 256, 1, 1),
        (1, 64, 28, 28),
        (0, 0, 0, 0),
        (1, 1),
    ),
    ConvBenchmark(
        "benchmark_conv_mobilenet_depthwise_3x3",
        (1, 32, 112, 112),
        (32, 1, 3, 3),
        (1, 32, 112, 112),
        (1, 1, 1, 1),
        (1, 1),
        groups=32,
    ),
    ConvBenchmark(
        "benchmark_conv_mobilenet_pointwise_1x1",
        (1, 32, 112, 112),
        (64, 32, 1, 1),
        (1, 64, 112, 112),
        (0, 0, 0, 0),
        (1, 1),
    ),
]


def conv2d(x: np.ndarray, w: np.ndarray, case: ConvBenchmark) -> np.ndarray:
	batch, in_ch, in_h, in_w = x.shape
	out_ch, in_ch_per_group, kernel_h, kernel_w = w.shape
	_, _, out_h, out_w = case.y_shape
	pad_top, pad_left, pad_bottom, pad_right = case.pads
	stride_h, stride_w = case.strides
	out = np.zeros(case.y_shape, dtype=np.float32)
	out_ch_per_group = out_ch // case.groups
	x_padded = np.pad(
		x,
		((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
		mode="constant",
	)

	for group in range(case.groups):
		c_start = group * in_ch_per_group
		m_start = group * out_ch_per_group
		x_group = x_padded[:, c_start:c_start + in_ch_per_group, :, :]
		w_group = w[m_start:m_start + out_ch_per_group, :, :, :]
		for ky in range(kernel_h):
			for kx in range(kernel_w):
				patch = x_group[
					:,
					:,
					ky:ky + out_h * stride_h:stride_h,
					kx:kx + out_w * stride_w:stride_w,
				]
				out[:, m_start:m_start + out_ch_per_group, :, :] += np.einsum(
					"bcyx,mc->bmyx",
					patch,
					w_group[:, :, ky, kx],
					optimize=True,
				)
	return out


def deterministic_array(shape: tuple[int, ...], offset: int) -> np.ndarray:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return ((values + offset) % 17) / np.float32(17.0)


def save_tensor(array: np.ndarray, path: Path) -> None:
    path.write_bytes(numpy_helper.from_array(array).SerializeToString())


def write_case(case: ConvBenchmark) -> None:
    out_dir = SCRIPT_DIR / case.name
    data_dir = out_dir / "test_data_set_0"
    data_dir.mkdir(parents=True, exist_ok=True)

    x = deterministic_array(case.x_shape, 1)
    w = deterministic_array(case.w_shape, 7)
    y = conv2d(x, w, case)

    node = helper.make_node(
        "Conv",
        ["X", "w"],
        ["Y"],
        name="sclbl-onnx-node1",
        kernel_shape=[case.w_shape[2], case.w_shape[3]],
        pads=list(case.pads),
        strides=list(case.strides),
        dilations=[1, 1],
        group=case.groups,
    )
    graph = helper.make_graph(
        [node],
        case.name,
        [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, case.x_shape),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, case.w_shape),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, case.y_shape)],
    )
    model = helper.make_model(graph, producer_name="onnx2c-benchmark-generator")
    onnx.save(model, out_dir / "model.onnx")

    save_tensor(x, data_dir / "input_0.pb")
    save_tensor(w, data_dir / "input_1.pb")
    save_tensor(y, data_dir / "output_0.pb")


def main() -> None:
    for case in CASES:
        write_case(case)


if __name__ == "__main__":
    main()
