#!/usr/bin/env python3
"""Create ONNX test models for im2col pass testing"""

import onnx
from onnx import helper, TensorProto
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_multi_conv_benchmark():
    """2-layer Conv benchmark model
    Input: (1, 3, 16, 16)
    Conv1: 3 -> 8, kernel 3x3, pad 1
    Conv2: 8 -> 8, kernel 3x3, pad 1
    Output: (1, 8, 16, 16)
    """
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 16, 16])
    
    W1 = helper.make_tensor('w1', TensorProto.FLOAT, [8, 3, 3, 3], vals=[1.0]*216)
    conv1 = helper.make_node('Conv', ['input', 'w1'], ['x1'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    
    W2 = helper.make_tensor('w2', TensorProto.FLOAT, [8, 8, 3, 3], vals=[1.0]*576)
    conv2 = helper.make_node('Conv', ['x1', 'w2'], ['output'], kernel_shape=[3, 3], pads=[1, 1, 1, 1])
    
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 8, 16, 16])
    
    graph = helper.make_graph([conv1, conv2], 'multi_conv', [X], [Y], [W1, W2])
    model = helper.make_model(graph, producer_name='benchmark')
    model.opset_import[0].version = 13
    return model

if __name__ == '__main__':
    model = create_multi_conv_benchmark()
    path = os.path.join(SCRIPT_DIR, 'multi_conv_bench.onnx')
    onnx.save(model, path)
    print(f"Created: multi_conv_bench.onnx")
