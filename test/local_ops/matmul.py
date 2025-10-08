import numpy as np
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model
import os

np.random.seed(0)

shapes = [
    ([2, 3], [3, 4]),
    # Elementwise broadcasting
    ([5, 2, 3], [5, 3, 4]),
    ([4, 5, 2, 3], [4, 5, 3, 4]),
    # Uneven Broadcasting 
    ([2, 3, 4], [4, 5]),
    ([3, 4], [2, 4, 5]),
    # Vectors
    ([3], [3]),
    ([3, 4], [4]),
    ([2, 3, 4], [4]),
    ([2, 3, 4, 5], [5]),
    ([3], [3, 4]),
    ([3], [2, 3, 4]),
]

for (a, b) in shapes:
    name = f"matmul_{'x'.join(map(str, a))}_{'x'.join(map(str, b))}"
    dir_name = "test_" + name
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(f"{dir_name}/test_data_set_0", exist_ok=True)

    a_array = np.random.rand(*a).astype(np.float32)
    b_array = np.random.rand(*b).astype(np.float32)
    y_array = np.matmul(a_array, b_array)

    A = make_tensor_value_info("A", onnx.TensorProto.FLOAT, a)
    B = make_tensor_value_info("B", onnx.TensorProto.FLOAT, b)
    Y = make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None] * len(y_array.shape))

    node = make_node("MatMul", ["A", "B"], ["Y"])

    graph = make_graph([node], name, [A, B], [Y])
    model = make_model(graph, producer_name="matmul.py")

    check_model(model)

    onnx.save(model, f"{dir_name}/model.onnx")

    def save_tensor(t, fn):
        with open(fn, 'wb') as f:
            npt = onnx.numpy_helper.from_array(t)
            f.write(npt.SerializeToString())

    save_tensor(a_array, f"{dir_name}/test_data_set_0/input_0.pb")
    save_tensor(b_array, f"{dir_name}/test_data_set_0/input_1.pb")
    save_tensor(y_array, f"{dir_name}/test_data_set_0/output_0.pb")

    print(f"local_node_test({name})")