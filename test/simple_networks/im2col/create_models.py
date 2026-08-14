#!/usr/bin/env python3
"""Create ONNX test models for im2col pass testing"""

import onnx
from onnx import helper, TensorProto
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_simple_conv_model():
    """Simple Conv: single conv layer, no bias
    Input: (1, 1, 5, 5)
    Conv: kernel 2x2, stride 1
    Output: (1, 1, 4, 4)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [1, 1, 2, 2],
        vals=[1.0, 1.0, 1.0, 1.0])

    conv_node = helper.make_node('Conv', inputs=['input', 'weight'], outputs=['output'],
        kernel_shape=[2, 2], strides=[1, 1], pads=[0, 0, 0, 0])

    graph = helper.make_graph([conv_node], 'simple_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def create_conv_with_bias_model():
    """Conv with bias (2 output channels)
    Input: (1, 1, 5, 5)
    Conv: kernel 2x2
    Output: (1, 2, 4, 4)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 4, 4])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [2, 1, 2, 2],
        vals=[1.0]*8)
    B = helper.make_tensor('bias', TensorProto.FLOAT, [2],
        vals=[0.0, 0.0])

    conv_node = helper.make_node('Conv', inputs=['input', 'weight', 'bias'],
        outputs=['output'], kernel_shape=[2, 2])

    graph = helper.make_graph([conv_node], 'conv_with_bias', [X], [Y], [W, B])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def create_conv_stride2_model():
    """Conv with stride=2
    Input: (1, 1, 7, 7)
    Conv: kernel 2x2, stride 2
    Output: (1, 1, 3, 3)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 7, 7])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 3, 3])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [1, 1, 2, 2],
        vals=[1.0, 2.0, 3.0, 4.0])

    conv_node = helper.make_node('Conv', inputs=['input', 'weight'], outputs=['output'],
        kernel_shape=[2, 2], strides=[2, 2])

    graph = helper.make_graph([conv_node], 'conv_stride2', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def create_depthwise_conv_model():
    """Depthwise Conv (group=2)
    Input: (1, 2, 4, 4)
    Conv: kernel 2x2, group 2
    Output: (1, 2, 3, 3)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 4, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 3, 3])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [2, 1, 2, 2],
        vals=[1.0]*4 + [2.0]*4)

    conv_node = helper.make_node('Conv', inputs=['input', 'weight'], outputs=['output'],
        kernel_shape=[2, 2], group=2)

    graph = helper.make_graph([conv_node], 'depthwise_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def create_pointwise_conv_model():
    """Pointwise Conv (1x1 kernel)
    Input: (1, 3, 4, 4)
    Conv: kernel 1x1
    Output: (1, 5, 4, 4)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5, 4, 4])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [5, 3, 1, 1],
        vals=[1.0]*15)

    conv_node = helper.make_node('Conv', inputs=['input', 'weight'], outputs=['output'],
        kernel_shape=[1, 1])

    graph = helper.make_graph([conv_node], 'pointwise_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def create_multi_channel_conv_model():
    """Multi-channel Conv (3 input, 4 output channels)
    Input: (1, 3, 4, 4)
    Conv: kernel 3x3
    Output: (1, 4, 2, 2)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4, 2, 2])

    W = helper.make_tensor('weight', TensorProto.FLOAT, [4, 3, 3, 3],
        vals=[float(i+1) for i in range(108)])

    conv_node = helper.make_node('Conv', inputs=['input', 'weight'], outputs=['output'],
        kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    graph = helper.make_graph([conv_node], 'multi_channel_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_im2col')
    model.opset_import[0].version = 13
    return model

def main():
    output_dir = SCRIPT_DIR

    models = [
        ('conv_simple', create_simple_conv_model()),
        ('conv_with_bias', create_conv_with_bias_model()),
        ('conv_stride2', create_conv_stride2_model()),
        ('depthwise_conv', create_depthwise_conv_model()),
        ('pointwise_conv', create_pointwise_conv_model()),
        ('multi_channel_conv', create_multi_channel_conv_model()),
    ]

    for name, model in models:
        path = os.path.join(output_dir, f'{name}.onnx')
        onnx.save(model, path)
        print(f"Created: {name}.onnx")

    print(f"\nCreated {len(models)} test models in {output_dir}")

if __name__ == '__main__':
    main()