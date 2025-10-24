import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import numpy as np

np.random.seed(0)
data = np.random.randn(20) * 10

data[0] = np.nan
data[1] = np.inf
data[2] = -np.inf
data[3] = 123
data[4] = -123
data[5] = np.pow(2.0, -24) # Smallest Subnormal
data[6] = np.pow(2.0, -14) # Smallest Normal
data[7] = np.pow(2.0, 15)  # Largest Normal
data[8] = -np.pow(2.0, -24) # Smallest Subnormal Negative
data[9] = -np.pow(2.0, -14) # Smallest Normal Negative
data[10] = -np.pow(2.0, 15)  # Largest Normal Negative
data[11] = np.pow(2.0, -30) # Smaller than Subnormal -> 0

data = data.astype(np.float16)

make_tensor_value_info("data", onnx.TensorProto.FLOAT16, [])
output = make_tensor_value_info("output", onnx.TensorProto.FLOAT, [])

node = make_node("Cast", inputs=["data"], outputs=["output"], to=onnx.TensorProto.FLOAT)

graph = make_graph([node], "float16_identity", [], [output])

graph.initializer.append(
    onnx.helper.make_tensor(
        name="data",
        data_type=onnx.TensorProto.FLOAT16,
        dims=data.shape,
        vals=data.tobytes(),
        raw=True,
    )
)

model = make_model(graph, producer_name="float16.py")
check_model(model)

onnx.save(model, "float16.onnx")

with open("float16.c", "w") as f:
    data = data.astype(np.float32)
    f.write("#include <math.h>\n\n")
    f.write(f"float result[{data.shape[0]}];\n")
    reference = ", ".join([str(x).replace("nan", "NAN").replace("inf", "INFINITY") for x in data])
    f.write(f"float reference[{data.shape[0]}] = {{{reference}}};\n")
    f.write(f"unsigned count = {data.shape[0]};\n")
