import os
import numpy as np
import onnx
from onnx import helper, numpy_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def qlinear_averagepool_ref(
    x: np.ndarray,
    x_scale: np.ndarray,
    x_zero_point: np.ndarray,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    kernel_shape,
    strides,
    pads,
    count_include_pad: int,
) -> np.ndarray:
    n, c, h, w = x.shape
    kh, kw = kernel_shape
    sh, sw = strides
    pt, pl, pb, pr = pads

    out_h = (h + pt + pb - kh) // sh + 1
    out_w = (w + pl + pr - kw) // sw + 1

    y = np.zeros((n, c, out_h, out_w), dtype=np.float64)
    xzp = int(x_zero_point.reshape(-1)[0])
    yzp = int(y_zero_point.reshape(-1)[0])
    scale = float(x_scale.reshape(-1)[0]) / float(y_scale.reshape(-1)[0])

    for ni in range(n):
        for ci in range(c):
            for oy in range(out_h):
                for ox in range(out_w):
                    h_start = oy * sh - pt
                    w_start = ox * sw - pl
                    total = 0
                    num = 0
                    for ky in range(kh):
                        iy = h_start + ky
                        for kx in range(kw):
                            ix = w_start + kx
                            if 0 <= iy < h and 0 <= ix < w:
                                total += int(x[ni, ci, iy, ix]) - xzp
                                num += 1
                    if count_include_pad:
                        num = kh * kw
                    value = (total * scale / float(num)) + yzp
                    y[ni, ci, oy, ox] = value

    if x.dtype == np.uint8:
        lo, hi, out_dtype = 0, 255, np.uint8
    elif x.dtype == np.int8:
        lo, hi, out_dtype = -128, 127, np.int8
    else:
        raise ValueError("Unsupported quantized type")

    y = round_away_from_zero(y)
    y = np.clip(y, lo, hi)
    return y.astype(out_dtype)


def save_tensor(t: np.ndarray, fn: str) -> None:
    with open(fn, "wb") as f:
        f.write(numpy_helper.from_array(t).SerializeToString())


def make_test(
    name,
    dtype_proto,
    x,
    x_scale,
    x_zero_point,
    y_scale,
    y_zero_point,
    kernel_shape,
    strides,
    pads,
    count_include_pad,
):
    dtype_np = onnx.helper.tensor_dtype_to_np_dtype(dtype_proto)

    x_scale = np.array(x_scale, dtype=np.float32)
    y_scale = np.array(y_scale, dtype=np.float32)
    x_zero_point = np.array(x_zero_point, dtype=dtype_np)
    y_zero_point = np.array(y_zero_point, dtype=dtype_np)

    y = qlinear_averagepool_ref(
        x=x,
        x_scale=x_scale,
        x_zero_point=x_zero_point,
        y_scale=y_scale,
        y_zero_point=y_zero_point,
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
    )

    x_info = helper.make_tensor_value_info("x", dtype_proto, list(x.shape))
    x_scale_info = helper.make_tensor_value_info("x_scale", onnx.TensorProto.FLOAT, [])
    x_zp_info = helper.make_tensor_value_info("x_zero_point", dtype_proto, [])
    y_scale_info = helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zp_info = helper.make_tensor_value_info("y_zero_point", dtype_proto, [])
    y_info = helper.make_tensor_value_info("y", dtype_proto, list(y.shape))

    node = helper.make_node(
        "QLinearAveragePool",
        ["x", "x_scale", "x_zero_point", "y_scale", "y_zero_point"],
        ["y"],
        domain="com.microsoft",
        kernel_shape=kernel_shape,
        strides=strides,
        pads=pads,
        count_include_pad=count_include_pad,
    )

    graph = helper.make_graph(
        [node],
        name,
        [x_info, x_scale_info, x_zp_info, y_scale_info, y_zp_info],
        [y_info],
    )
    model = helper.make_model(
        graph,
        producer_name="qlinearaveragepool.py",
        opset_imports=[helper.make_opsetid("com.microsoft", 1)],
    )
    onnx.checker.check_model(model)

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
    name="qlinearaveragepool_uint8_basic",
    dtype_proto=onnx.TensorProto.UINT8,
    x=np.array([[[[130, 131, 132], [129, 128, 127], [126, 125, 124]]]], dtype=np.uint8),
    x_scale=0.25,
    x_zero_point=128,
    y_scale=0.20,
    y_zero_point=127,
    kernel_shape=[2, 2],
    strides=[1, 1],
    pads=[0, 0, 0, 0],
    count_include_pad=0,
)

make_test(
    name="qlinearaveragepool_uint8_padded_exclude_pad",
    dtype_proto=onnx.TensorProto.UINT8,
    x=np.array([[[[140, 145, 150], [135, 130, 125], [120, 115, 110]]]], dtype=np.uint8),
    x_scale=0.10,
    x_zero_point=128,
    y_scale=0.05,
    y_zero_point=120,
    kernel_shape=[2, 2],
    strides=[2, 2],
    pads=[1, 1, 1, 1],
    count_include_pad=0,
)

make_test(
    name="qlinearaveragepool_int8_padded_include_pad",
    dtype_proto=onnx.TensorProto.INT8,
    x=np.array([[[[-3, -1, 2], [4, -5, 6], [7, 0, -2]]]], dtype=np.int8),
    x_scale=0.50,
    x_zero_point=0,
    y_scale=0.25,
    y_zero_point=-3,
    kernel_shape=[2, 2],
    strides=[2, 2],
    pads=[1, 1, 1, 1],
    count_include_pad=1,
)
