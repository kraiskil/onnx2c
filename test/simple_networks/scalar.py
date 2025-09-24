import numpy as np
import tensorflow as tf
import tf2onnx

sig = [tf.TensorSpec([], tf.float32, name="a")]

@tf.function(input_signature=sig)
def scalar_identity(a):
    return tf.identity(a)

@tf.function(input_signature=sig)
def scalar_abs(a):
    return tf.abs(a)

@tf.function(input_signature=sig)
def scalar_add(a):
    return a + tf.constant(1, dtype=tf.float32)

for func in [scalar_identity, scalar_abs, scalar_add]:
    tf2onnx.convert.from_function(
        func,
        input_signature=func.input_signature,
        opset=13,
        output_path=f"{func.__name__}.onnx"
    )