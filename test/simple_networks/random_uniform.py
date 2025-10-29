import numpy as np
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

shape = [3, 4, 5]

output = make_tensor_value_info("output", onnx.TensorProto.FLOAT, shape)

node = make_node(
    "RandomUniform",
    inputs=[],
    outputs=["output"],
    shape=shape,
    dtype=onnx.TensorProto.FLOAT,
    high=1.0,
    low=0.0
)

graph = make_graph([node], "random_uniform", [], [output])
model = make_model(graph, producer_name="random_uniform.py")

check_model(model)

onnx.save(model, "random_uniform.onnx")
