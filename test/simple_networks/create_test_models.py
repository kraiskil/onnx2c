#!/usr/bin/env python3
"""Create simple ONNX test models for conv operations"""

import onnx
from onnx import helper, TensorProto
import os

def create_simple_conv_model():
    """Simple Conv: single conv layer, no bias"""
    # Input: (1, 1, 5, 5)
    # Conv: kernel 2x2, stride 1
    # Output: (1, 1, 4, 4)
    
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 4, 4])
    
    W = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        [1, 1, 2, 2],
        vals=[1.0, 1.0, 1.0, 1.0]
    )
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[1, 1],
        pads=[0, 0, 0, 0]
    )
    
    graph = helper.make_graph([conv_node], 'simple_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_conv')
    model.opset_import[0].version = 13
    return model

def create_conv_with_bias_model():
    """Conv with bias"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 5, 5])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 4, 4])
    
    W = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        [2, 1, 2, 2],
        vals=[1.0]*8
    )
    
    B = helper.make_tensor(
        'bias',
        TensorProto.FLOAT,
        [2],
        vals=[0.0, 0.0]
    )
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight', 'bias'],
        outputs=['output'],
        kernel_shape=[2, 2]
    )
    
    graph = helper.make_graph([conv_node], 'conv_with_bias', [X], [Y], [W, B])
    model = helper.make_model(graph, producer_name='test_conv')
    model.opset_import[0].version = 13
    return model

def create_conv_stride2_model():
    """Conv with stride=2"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 1, 7, 7])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 1, 3, 3])
    
    W = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        [1, 1, 2, 2],
        vals=[1.0, 2.0, 3.0, 4.0]
    )
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight'],
        outputs=['output'],
        kernel_shape=[2, 2],
        strides=[2, 2]
    )
    
    graph = helper.make_graph([conv_node], 'conv_stride2', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_conv')
    model.opset_import[0].version = 13
    return model

def create_depthwise_conv_model():
    """Depthwise Conv"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2, 4, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2, 3, 3])
    
    W = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        [2, 1, 2, 2],
        vals=[1.0]*4 + [2.0]*4
    )
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight'],
        outputs=['output'],
        kernel_shape=[2, 2],
        group=2
    )
    
    graph = helper.make_graph([conv_node], 'depthwise_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_conv')
    model.opset_import[0].version = 13
    return model

def create_pointwise_conv_model():
    """Pointwise Conv (1x1)"""
    X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5, 4, 4])
    
    W = helper.make_tensor(
        'weight',
        TensorProto.FLOAT,
        [5, 3, 1, 1],
        vals=[1.0]*15
    )
    
    conv_node = helper.make_node(
        'Conv',
        inputs=['input', 'weight'],
        outputs=['output'],
        kernel_shape=[1, 1]
    )
    
    graph = helper.make_graph([conv_node], 'pointwise_conv', [X], [Y], [W])
    model = helper.make_model(graph, producer_name='test_conv')
    model.opset_import[0].version = 13
    return model

# Save all models
output_dir = '/Users/michaelschirmer/Documents/Code/onnx2c/test/simple_networks'
os.makedirs(output_dir, exist_ok=True)

models = [
    ('test_conv_simple.onnx', create_simple_conv_model()),
    ('test_conv_with_bias.onnx', create_conv_with_bias_model()),
    ('test_conv_stride2.onnx', create_conv_stride2_model()),
    ('test_depthwise_conv.onnx', create_depthwise_conv_model()),
    ('test_pointwise_conv.onnx', create_pointwise_conv_model()),
]

for name, model in models:
    path = os.path.join(output_dir, name)
    onnx.save(model, path)
    print(f"Saved: {name}")

print("\nAll test models created!")