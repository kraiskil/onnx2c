import os
import numpy as np
import onnx
from onnx import helper, numpy_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def qlinear_softmax_ref(x, x_scale, x_zero_point, y_scale, y_zero_point, axis=-1):
    x_scale = float(np.array(x_scale, dtype=np.float32).reshape(-1)[0])
    y_scale = float(np.array(y_scale, dtype=np.float32).reshape(-1)[0])
    xzp = int(np.array(x_zero_point).reshape(-1)[0])
    yzp = int(np.array(y_zero_point).reshape(-1)[0])

    x_deq = x_scale * (x.astype(np.int32) - xzp)
    maxv = np.max(x_deq, axis=axis, keepdims=True)
    expv = np.exp(x_deq - maxv)
    p = expv / np.sum(expv, axis=axis, keepdims=True)

    y = round_away_from_zero(p / y_scale + yzp)

    if x.dtype == np.uint8:
        lo, hi, out_dtype = 0, 255, np.uint8
    elif x.dtype == np.int8:
        lo, hi, out_dtype = -128, 127, np.int8
    else:
        raise ValueError("Unsupported quantized dtype")

    y = np.clip(y, lo, hi)
    return y.astype(out_dtype)


def save_tensor(t, fn):
    with open(fn, "wb") as f:
        f.write(numpy_helper.from_array(np.array(t)).SerializeToString())


def make_test(name, dtype_proto, x, x_scale, x_zero_point, y_scale, y_zero_point, axis_attr=None):
    dtype_np = onnx.helper.tensor_dtype_to_np_dtype(dtype_proto)
    x = np.array(x, dtype=dtype_np)
    x_scale = np.array(x_scale, dtype=np.float32)
    x_zero_point = np.array(x_zero_point, dtype=dtype_np)
    y_scale = np.array(y_scale, dtype=np.float32)
    y_zero_point = np.array(y_zero_point, dtype=dtype_np)

    axis = axis_attr if axis_attr is not None else -1
    y = qlinear_softmax_ref(x, x_scale, x_zero_point, y_scale, y_zero_point, axis=axis)

    x_info = helper.make_tensor_value_info("x", dtype_proto, list(x.shape))
    x_scale_info = helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zp_info = helper.make_tensor_value_info("x_zero_point", dtype_proto, [])
    y_scale_info = helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zp_info = helper.make_tensor_value_info("y_zero_point", dtype_proto, [])
    y_info = helper.make_tensor_value_info("y", dtype_proto, list(y.shape))

    attrs = {}
    if axis_attr is not None:
        attrs["axis"] = axis_attr

    node = helper.make_node(
        "QLinearSoftmax",
        ["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        domain="com.microsoft",
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        name,
        [x_info, x_scale_info, x_zp_info, y_scale_info, y_zp_info],
        [y_info],
    )

    model = helper.make_model(
        graph,
        producer_name="qlinearsoftmax.py",
        opset_imports=[helper.make_opsetid("com.microsoft", 1)],
    )

    try:
        onnx.checker.check_model(model)
    except Exception:
        pass

    test_dir = os.path.join(BASE_DIR, f"test_{name}")
    data_dir = os.path.join(test_dir, "test_data_set_0")
    os.makedirs(data_dir, exist_ok=True)

    onnx.save(model, os.path.join(test_dir, "model.onnx"))

    inputs = [x, x_scale, x_zero_point, y_scale, y_zero_point]
    for i, t in enumerate(inputs):
        save_tensor(t, os.path.join(data_dir, f"input_{i}.pb"))
    save_tensor(y, os.path.join(data_dir, "output_0.pb"))

    print(f"local_node_test({name})")


make_test(
    name="qlinearsoftmax_uint8_2d",
    dtype_proto=onnx.TensorProto.UINT8,
    x=[[120, 123, 130, 140], [118, 125, 127, 126]],
    x_scale=0.05,
    x_zero_point=128,
    y_scale=1.0 / 256.0,
    y_zero_point=0,
    axis_attr=1,
)

make_test(
    name="qlinearsoftmax_int8_2d",
    dtype_proto=onnx.TensorProto.INT8,
    x=[[-8, -2, 1, 4], [3, -1, -6, 2]],
    x_scale=0.10,
    x_zero_point=0,
    y_scale=1.0 / 256.0,
    y_zero_point=-128,
    axis_attr=1,
)

make_test(
    name="qlinearsoftmax_uint8_3d_last_axis",
    dtype_proto=onnx.TensorProto.UINT8,
    x=[
        [[130, 128, 126], [127, 140, 120]],
        [[125, 124, 132], [129, 131, 128]],
    ],
    x_scale=0.04,
    x_zero_point=128,
    y_scale=1.0 / 256.0,
    y_zero_point=0,
    axis_attr=-1,
)
