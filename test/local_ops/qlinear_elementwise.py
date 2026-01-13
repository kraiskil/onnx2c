import numpy as np
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import onnxruntime as ort
import os

np.random.seed(0)

CONFIGS = {
    "int8": {
        "type": onnx.TensorProto.INT8,
        "zero_point": 0
    },
    "uint8": {
        "type": onnx.TensorProto.UINT8,
        "zero_point": 128
    }
}

def random_tensor(tensor_info):
    shape = [dim.dim_value for dim in tensor_info.type.tensor_type.shape.dim]
    if tensor_info.type.tensor_type.elem_type == onnx.TensorProto.INT8:
        return np.random.randint(-128, 127, size=shape, dtype=np.int8)
    elif tensor_info.type.tensor_type.elem_type == onnx.TensorProto.UINT8:
        return np.random.randint(0, 255, size=shape, dtype=np.uint8)
    else:
        raise ValueError("Unsupported tensor type")

for op_name in ["QLinearAdd", "QLinearMul"]:
    for config_name, config in CONFIGS.items():
        name = op_name.lower() + "_" + config_name
        dir_name = "test_" + name

        data_shape = (2, 3, 4)

        a = make_tensor_value_info("a", config["type"], data_shape)
        a_scale = make_tensor_value_info("a_scale", onnx.TensorProto.FLOAT, ())
        a_zero_point = make_tensor_value_info("a_zero_point", config["type"], ())

        b = make_tensor_value_info("b", config["type"], data_shape)
        b_scale = make_tensor_value_info("b_scale", onnx.TensorProto.FLOAT, ())
        b_zero_point = make_tensor_value_info("b_zero_point", config["type"], ())

        c_scale = make_tensor_value_info("c_scale", onnx.TensorProto.FLOAT, ())
        c_zero_point = make_tensor_value_info("c_zero_point", config["type"], ())

        c = make_tensor_value_info("c", config["type"], data_shape)

        node = make_node(op_name, [
            "a", "a_scale", "a_zero_point",
            "b", "b_scale", "b_zero_point",
            "c_scale", "c_zero_point"
        ], ["c"], domain="com.microsoft")

        graph = make_graph([node], name, [
            a, a_scale, a_zero_point,
            b, b_scale, b_zero_point,
            c_scale, c_zero_point
        ], [c])

        model = make_model(graph,
            producer_name="qlinear_elementwise.py",
            opset_imports=[onnx.helper.make_opsetid("com.microsoft", 1)]
        )

        check_model(model)

        dtype = onnx.helper.tensor_dtype_to_np_dtype(config["type"])
        
        inputs = {
            "a": random_tensor(a),
            "b": random_tensor(b),

            "a_zero_point": np.array(config["zero_point"], dtype=dtype),
            "b_zero_point": np.array(config["zero_point"], dtype=dtype),
            "c_zero_point": np.array(config["zero_point"], dtype=dtype),

            "a_scale": np.array(np.random.rand() * 0.1).astype(np.float32),
            "b_scale": np.array(np.random.rand() * 0.1).astype(np.float32),
            "c_scale": np.array(np.random.rand() * 0.1).astype(np.float32)
        }

        sess = ort.InferenceSession(model.SerializeToString())
        outputs = sess.run(["c"], inputs)

        def save_tensor(t, fn):
            with open(fn, 'wb') as f:
                npt = onnx.numpy_helper.from_array(t)
                f.write(npt.SerializeToString())

        os.makedirs(dir_name, exist_ok=True)
        os.makedirs(f"{dir_name}/test_data_set_0", exist_ok=True)

        onnx.save(model, f"{dir_name}/model.onnx")

        for index, input_tensor in enumerate(graph.input):
            save_tensor(inputs[input_tensor.name], f"{dir_name}/test_data_set_0/input_{index}.pb")

        save_tensor(outputs[0], f"{dir_name}/test_data_set_0/output_0.pb")

        print(f"local_node_test({name})")