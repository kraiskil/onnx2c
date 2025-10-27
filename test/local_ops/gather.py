import numpy as np
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import onnxruntime as ort
import os

np.random.seed(0)

tests = {
    "gather_basic": {
        "data": (2, 3, 4),
        "indices": (4, 5),
        "axis": 1
    },
    "gather_scalar_axis0": {
        "data": (2, 3, 4),
        "indices": (),
        "axis": 0
    },
    "gather_scalar_axis1": {
        "data": (2, 3, 4),
        "indices": (),
        "axis": 1
    },
    "gather_output_scalar": {
        "data": (10,),
        "indices": (),
        "axis": 0
    }
}

for name, config in tests.items():
    dir_name = "test_" + name

    data_shape = config["data"]
    indices_shape = config["indices"]
    axis = config["axis"]

    output_shape = tuple(data_shape[0:axis]) + tuple(indices_shape) + tuple(data_shape[axis+1:])

    data = make_tensor_value_info("data", onnx.TensorProto.FLOAT, data_shape)
    indices = make_tensor_value_info("indices", onnx.TensorProto.INT32, indices_shape)
    output = make_tensor_value_info("output", onnx.TensorProto.FLOAT, output_shape)

    node = make_node("Gather", ["data", "indices"], ["output"], axis=axis)

    graph = make_graph([node], name, [data, indices], [output])
    model = make_model(graph, producer_name="gather.py")

    check_model(model)

    inputs = {
        "data": np.random.rand(*data_shape).astype(np.float32),
        "indices": np.random.randint(0, data_shape[axis], indices_shape, dtype=np.int32)
    }

    sess = ort.InferenceSession(model.SerializeToString())
    outputs = sess.run(["output"], inputs)

    def save_tensor(t, fn):
        with open(fn, 'wb') as f:
            npt = onnx.numpy_helper.from_array(t)
            f.write(npt.SerializeToString())

    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(f"{dir_name}/test_data_set_0", exist_ok=True)

    onnx.save(model, f"{dir_name}/model.onnx")

    save_tensor(inputs["data"], f"{dir_name}/test_data_set_0/input_0.pb")
    save_tensor(inputs["indices"], f"{dir_name}/test_data_set_0/input_1.pb")
    save_tensor(outputs[0], f"{dir_name}/test_data_set_0/output_0.pb")

    print(f"local_node_test({name})")