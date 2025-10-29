import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import numpy as np

configs = {
    "float16": (
        onnx.TensorProto.FLOAT16,
        lambda x: x.astype(np.float16),
        5
    ),
    "bfloat16": (
        onnx.TensorProto.BFLOAT16,
        # Numpy does not have bfloat16 type, so we round by zeroing the lower 16 bits
        # this ensures that the float32 value can be exactly represented as bfloat16
        lambda x: (x.astype(np.float32).view(np.uint32) >> 16 << 16).view(np.float32),
        8
    ),
}

for name, (onnx_type, round_func, exponent_size) in configs.items():
    np.random.seed(0)
    data = np.random.randn(20) * 10

    bias = (1 << (exponent_size - 1)) - 1
    max_exponent = (1 << exponent_size) - 2 - bias
    min_exponent = 1 - bias

    data[0] = np.nan
    data[1] = np.inf
    data[2] = -np.inf
    data[3] = 123
    data[4] = -123
    data[5] = np.pow(2.0, min_exponent - 1) # Subnormal
    data[6] = np.pow(2.0, min_exponent) # Smallest Normal
    data[7] = np.pow(2.0, max_exponent)  # Largest Normal
    data[8] = -np.pow(2.0, min_exponent - 1) # Subnormal Negative
    data[9] = -np.pow(2.0, min_exponent) # Smallest Normal Negative
    data[10] = -np.pow(2.0, max_exponent)  # Largest Normal Negative

    data = round_func(data)

    make_tensor_value_info("data", onnx_type, [])
    output = make_tensor_value_info("output", onnx.TensorProto.FLOAT, [])

    node = make_node("Cast", inputs=["data"], outputs=["output"], to=onnx.TensorProto.FLOAT)

    graph = make_graph([node], name, [], [output])

    graph.initializer.append(
        onnx.helper.make_tensor(
            name="data",
            data_type=onnx_type,
            dims=data.shape,
            vals=data,
            raw=False,
        )
    )

    model = make_model(graph, producer_name="fp.py")
    check_model(model)

    onnx.save(model, f"fp_{name}.onnx")

    with open(f"fp_{name}.c", "w") as f:
        data = data.astype(np.float32)
        f.write("#include <math.h>\n\n")
        f.write(f"float result[{data.shape[0]}];\n")
        reference = ", ".join([str(x).replace("nan", "NAN").replace("inf", "INFINITY") for x in data])
        f.write(f"float reference[{data.shape[0]}] = {{{reference}}};\n")
        f.write(f"unsigned count = {data.shape[0]};\n")
    
    print(f"simple_test_fp( fp_{name} )")
