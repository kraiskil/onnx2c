#!/usr/bin/env python3
"""Generate one-node ONNX models for im2col benchmarking."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import onnx
from onnx import TensorProto, helper


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class ConvCase:
    name: str
    in_channels: int
    out_channels: int
    height: int
    width: int
    kernel: int
    stride: int = 1
    pad: int = 0
    dilation: int = 1
    group: int = 1
    bias: bool = False


def repeated_floats(count: int, scale: float = 0.01) -> list[float]:
    return [((i % 17) + 1) * scale for i in range(count)]


def repeated_ints(count: int, modulus: int = 251) -> list[int]:
    return [(i * 7 + 3) % modulus for i in range(count)]


def conv_output_dim(input_size: int, kernel: int, stride: int, pad: int, dilation: int) -> int:
    filter_size = kernel + (kernel - 1) * (dilation - 1)
    return (input_size + 2 * pad - filter_size) // stride + 1


def make_conv(case: ConvCase) -> onnx.ModelProto:
    out_h = conv_output_dim(case.height, case.kernel, case.stride, case.pad, case.dilation)
    out_w = conv_output_dim(case.width, case.kernel, case.stride, case.pad, case.dilation)
    in_per_group = case.in_channels // case.group

    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, case.in_channels, case.height, case.width])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, case.out_channels, out_h, out_w])

    weight_count = case.out_channels * in_per_group * case.kernel * case.kernel
    w = helper.make_tensor(
        "weight",
        TensorProto.FLOAT,
        [case.out_channels, in_per_group, case.kernel, case.kernel],
        vals=repeated_floats(weight_count),
    )
    initializers = [w]
    inputs = ["input", "weight"]

    if case.bias:
        b = helper.make_tensor("bias", TensorProto.FLOAT, [case.out_channels], vals=repeated_floats(case.out_channels))
        initializers.append(b)
        inputs.append("bias")

    conv = helper.make_node(
        "Conv",
        inputs=inputs,
        outputs=["output"],
        kernel_shape=[case.kernel, case.kernel],
        strides=[case.stride, case.stride],
        pads=[case.pad, case.pad, case.pad, case.pad],
        dilations=[case.dilation, case.dilation],
        group=case.group,
    )
    graph = helper.make_graph([conv], case.name, [x], [y], initializers)
    model = helper.make_model(graph, producer_name="onnx2c_im2col_benchmarks")
    model.opset_import[0].version = 13
    return model


def make_convinteger(case: ConvCase, with_zero_points: bool) -> onnx.ModelProto:
    out_h = conv_output_dim(case.height, case.kernel, case.stride, case.pad, case.dilation)
    out_w = conv_output_dim(case.width, case.kernel, case.stride, case.pad, case.dilation)

    x = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, case.in_channels, case.height, case.width])
    y = helper.make_tensor_value_info("output", TensorProto.INT32, [1, case.out_channels, out_h, out_w])
    weight_count = case.out_channels * case.in_channels * case.kernel * case.kernel
    w = helper.make_tensor(
        "weight",
        TensorProto.UINT8,
        [case.out_channels, case.in_channels, case.kernel, case.kernel],
        vals=repeated_ints(weight_count),
    )
    initializers = [w]
    inputs = ["input", "weight"]

    if with_zero_points:
        x_zero = helper.make_tensor("x_zero_point", TensorProto.UINT8, [], vals=[2])
        w_zero = helper.make_tensor("w_zero_point", TensorProto.UINT8, [], vals=[3])
        initializers.extend([x_zero, w_zero])
        inputs.extend(["x_zero_point", "w_zero_point"])

    conv = helper.make_node(
        "ConvInteger",
        inputs=inputs,
        outputs=["output"],
        kernel_shape=[case.kernel, case.kernel],
        strides=[case.stride, case.stride],
        pads=[case.pad, case.pad, case.pad, case.pad],
    )
    graph = helper.make_graph([conv], case.name, [x], [y], initializers)
    model = helper.make_model(graph, producer_name="onnx2c_im2col_benchmarks")
    model.opset_import[0].version = 13
    return model


def make_qlinearconv(case: ConvCase) -> onnx.ModelProto:
    out_h = conv_output_dim(case.height, case.kernel, case.stride, case.pad, case.dilation)
    out_w = conv_output_dim(case.width, case.kernel, case.stride, case.pad, case.dilation)
    in_per_group = case.in_channels // case.group

    x = helper.make_tensor_value_info("input", TensorProto.UINT8, [1, case.in_channels, case.height, case.width])
    y = helper.make_tensor_value_info("output", TensorProto.UINT8, [1, case.out_channels, out_h, out_w])

    weight_count = case.out_channels * in_per_group * case.kernel * case.kernel
    initializers = [
        helper.make_tensor("x_scale", TensorProto.FLOAT, [], vals=[0.02]),
        helper.make_tensor("x_zero_point", TensorProto.UINT8, [], vals=[2]),
        helper.make_tensor("weight", TensorProto.UINT8, [case.out_channels, in_per_group, case.kernel, case.kernel], vals=repeated_ints(weight_count)),
        helper.make_tensor("w_scale", TensorProto.FLOAT, [], vals=[0.03]),
        helper.make_tensor("w_zero_point", TensorProto.UINT8, [], vals=[3]),
        helper.make_tensor("y_scale", TensorProto.FLOAT, [], vals=[0.04]),
        helper.make_tensor("y_zero_point", TensorProto.UINT8, [], vals=[4]),
    ]

    inputs = ["input", "x_scale", "x_zero_point", "weight", "w_scale", "w_zero_point", "y_scale", "y_zero_point"]
    conv = helper.make_node(
        "QLinearConv",
        inputs=inputs,
        outputs=["output"],
        kernel_shape=[case.kernel, case.kernel],
        strides=[case.stride, case.stride],
        pads=[case.pad, case.pad, case.pad, case.pad],
        dilations=[case.dilation, case.dilation],
        group=case.group,
    )
    graph = helper.make_graph([conv], case.name, [x], [y], initializers)
    model = helper.make_model(graph, producer_name="onnx2c_im2col_benchmarks")
    model.opset_import[0].version = 13
    return model


def write_model(name: str, model: onnx.ModelProto) -> None:
    path = os.path.join(SCRIPT_DIR, f"{name}.onnx")
    onnx.save(model, path)
    print(f"wrote {path}")


def main() -> None:
    conv_cases: Iterable[ConvCase] = [
        ConvCase("conv_f32_k1_pointwise_8", 16, 32, 8, 8, 1),
        ConvCase("conv_f32_k1_pointwise_32", 16, 32, 32, 32, 1),
        ConvCase("conv_f32_k1_pointwise_64", 16, 32, 64, 64, 1),
        ConvCase("conv_f32_k1_pointwise_128", 16, 32, 128, 128, 1),
        ConvCase("conv_f32_k1_pointwise_192", 16, 32, 192, 192, 1),
        ConvCase("conv_f32_k1_pointwise_256", 16, 32, 256, 256, 1),
        ConvCase("conv_f32_k3_s1_pad1_16", 16, 16, 16, 16, 3, pad=1),
        ConvCase("conv_f32_k3_s1_pad1_32", 16, 16, 32, 32, 3, pad=1),
        ConvCase("conv_f32_k3_s1_pad1_64", 16, 16, 64, 64, 3, pad=1),
        ConvCase("conv_f32_k3_s2_pad1_64", 8, 16, 64, 64, 3, stride=2, pad=1),
        ConvCase("conv_f32_k3_s2_pad1_128", 8, 16, 128, 128, 3, stride=2, pad=1),
        ConvCase("conv_f32_k3_s1_pad1_128", 8, 16, 128, 128, 3, pad=1),
        ConvCase("conv_f32_k3_s1_pad1_192", 8, 16, 192, 192, 3, pad=1),
        ConvCase("conv_f32_k3_s1_pad1_256", 8, 16, 256, 256, 3, pad=1),
        ConvCase("conv_f32_k3_s2_pad1_224", 3, 16, 224, 224, 3, stride=2, pad=1),
        ConvCase("conv_f32_grouped_k3_g4_32", 16, 16, 32, 32, 3, pad=1, group=4),
        ConvCase("conv_f32_grouped_k3_g4_s2_64", 16, 16, 64, 64, 3, stride=2, pad=1, group=4),
        ConvCase("conv_f32_k5_s1_pad2_32", 8, 8, 32, 32, 5, pad=2, bias=True),
        ConvCase("conv_f32_k7_s1_pad3_32", 8, 8, 32, 32, 7, pad=3, bias=True),
        ConvCase("conv_f32_k3_d2_pad2_32", 8, 8, 32, 32, 3, pad=2, dilation=2),
        ConvCase("conv_f32_depthwise_k3_32", 16, 16, 32, 32, 3, pad=1, group=16),
        ConvCase("conv_f32_depthwise_k3_s2_64", 16, 16, 64, 64, 3, stride=2, pad=1, group=16),
        ConvCase("conv_f32_depthwise_k3_128", 16, 16, 128, 128, 3, pad=1, group=16),
    ]
    for case in conv_cases:
        write_model(case.name, make_conv(case))

    convinteger_cases = [
        (ConvCase("convinteger_u8_k1_pointwise_32", 16, 32, 32, 32, 1), False),
        (ConvCase("convinteger_u8_k3_s1_pad1_32", 8, 8, 32, 32, 3, pad=1), False),
        (ConvCase("convinteger_u8_k3_s1_pad1_64", 8, 8, 64, 64, 3, pad=1), False),
        (ConvCase("convinteger_u8_k3_s2_pad1_64", 8, 16, 64, 64, 3, stride=2, pad=1), False),
        (ConvCase("convinteger_u8_k3_s2_pad1_64_zp", 8, 16, 64, 64, 3, stride=2, pad=1), True),
        (ConvCase("convinteger_u8_k3_s1_pad1_128", 8, 8, 128, 128, 3, pad=1), False),
        (ConvCase("convinteger_u8_k3_s1_pad1_192", 8, 8, 192, 192, 3, pad=1), False),
        (ConvCase("convinteger_u8_k3_s1_pad1_256", 8, 8, 256, 256, 3, pad=1), False),
        (ConvCase("convinteger_u8_k5_s1_pad2_16_zp", 8, 8, 16, 16, 5, pad=2), True),
        (ConvCase("convinteger_u8_k5_s1_pad2_32_zp", 8, 8, 32, 32, 5, pad=2), True),
    ]
    for case, with_zero_points in convinteger_cases:
        write_model(case.name, make_convinteger(case, with_zero_points))

    qlinearconv_cases = [
        ConvCase("qlinearconv_u8_k1_pointwise_8", 16, 32, 8, 8, 1),
        ConvCase("qlinearconv_u8_k1_pointwise_32", 16, 32, 32, 32, 1),
        ConvCase("qlinearconv_u8_k1_pointwise_64", 16, 32, 64, 64, 1),
        ConvCase("qlinearconv_u8_k1_pointwise_128", 16, 32, 128, 128, 1),
        ConvCase("qlinearconv_u8_k1_pointwise_192", 16, 32, 192, 192, 1),
        ConvCase("qlinearconv_u8_k1_pointwise_256", 16, 32, 256, 256, 1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_16", 8, 8, 16, 16, 3, pad=1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_32", 8, 8, 32, 32, 3, pad=1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_64", 8, 8, 64, 64, 3, pad=1),
        ConvCase("qlinearconv_u8_k3_s2_pad1_64", 8, 16, 64, 64, 3, stride=2, pad=1),
        ConvCase("qlinearconv_u8_k3_s2_pad1_128", 8, 16, 128, 128, 3, stride=2, pad=1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_128", 8, 8, 128, 128, 3, pad=1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_192", 8, 8, 192, 192, 3, pad=1),
        ConvCase("qlinearconv_u8_k3_s1_pad1_256", 8, 8, 256, 256, 3, pad=1),
        ConvCase("qlinearconv_u8_grouped_k3_g4_32", 16, 16, 32, 32, 3, pad=1, group=4),
        ConvCase("qlinearconv_u8_grouped_k3_g4_s2_64", 16, 16, 64, 64, 3, stride=2, pad=1, group=4),
        ConvCase("qlinearconv_u8_depthwise_k3_32", 8, 8, 32, 32, 3, pad=1, group=8),
        ConvCase("qlinearconv_u8_depthwise_k3_64", 8, 8, 64, 64, 3, pad=1, group=8),
        ConvCase("qlinearconv_u8_depthwise_k3_128", 8, 8, 128, 128, 3, pad=1, group=8),
    ]
    for case in qlinearconv_cases:
        write_model(case.name, make_qlinearconv(case))


if __name__ == "__main__":
    main()
