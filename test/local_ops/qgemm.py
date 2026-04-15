import os
import numpy as np
import onnx
from onnx import helper, numpy_helper

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def round_away_from_zero(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def qgemm_ref(
    A: np.ndarray,
    a_scale: np.ndarray,
    a_zero_point: np.ndarray,
    B: np.ndarray,
    b_scale: np.ndarray,
    b_zero_point: np.ndarray,
    C: np.ndarray,
    y_scale: np.ndarray,
    y_zero_point: np.ndarray,
    transB: int,
) -> np.ndarray:
    M, K = A.shape
    N = B.shape[0] if transB else B.shape[1]

    azp = int(a_zero_point.reshape(-1)[0])
    bzp = int(b_zero_point.reshape(-1)[0])
    yzp = int(y_zero_point.reshape(-1)[0])
    scale = float(a_scale.reshape(-1)[0]) * float(b_scale.reshape(-1)[0]) / float(y_scale.reshape(-1)[0])

    c_arr = np.array(C, dtype=np.int32)

    y = np.zeros((M, N), dtype=np.float64)
    for r in range(M):
        for c in range(N):
            if c_arr.ndim == 0:
                bias = int(c_arr)
            elif c_arr.ndim == 1:
                dim = c_arr.shape[0]
                if dim == M:
                    bias = int(c_arr[r])
                elif dim == N:
                    bias = int(c_arr[c])
                elif dim == 1:
                    bias = int(c_arr[0])
                else:
                    raise ValueError("C dimension mismatch")
            elif c_arr.ndim == 2:
                rr = 0 if c_arr.shape[0] <= 1 else r
                cc = 0 if c_arr.shape[1] <= 1 else c
                bias = int(c_arr[rr, cc])
            else:
                raise ValueError("C rank too high")

            acc = bias
            for i in range(K):
                b_val = int(B[c, i]) if transB else int(B[i, c])
                acc += (int(A[r, i]) - azp) * (b_val - bzp)

            y[r, c] = float(acc) * scale + yzp

    if y_zero_point.dtype == np.uint8:
        lo, hi, dtype = 0, 255, np.uint8
    elif y_zero_point.dtype == np.int8:
        lo, hi, dtype = -128, 127, np.int8
    else:
        raise ValueError("Unsupported output dtype")

    y = round_away_from_zero(y)
    y = np.clip(y, lo, hi)
    return y.astype(dtype)


def save_tensor(t: np.ndarray, fn: str) -> None:
    with open(fn, "wb") as f:
        f.write(numpy_helper.from_array(np.array(t)).SerializeToString())


def make_test(
    name,
    dtype_proto,
    A,
    a_scale,
    a_zero_point,
    B,
    b_scale,
    b_zero_point,
    C,
    y_scale,
    y_zero_point,
    transB=0,
):
    dtype_np = onnx.helper.tensor_dtype_to_np_dtype(dtype_proto)

    A = np.array(A, dtype=dtype_np)
    B = np.array(B, dtype=dtype_np)
    C = np.array(C, dtype=np.int32)
    a_scale = np.array(a_scale, dtype=np.float32)
    b_scale = np.array(b_scale, dtype=np.float32)
    y_scale = np.array(y_scale, dtype=np.float32)
    a_zero_point = np.array(a_zero_point, dtype=dtype_np)
    b_zero_point = np.array(b_zero_point, dtype=dtype_np)
    y_zero_point = np.array(y_zero_point, dtype=dtype_np)

    Y = qgemm_ref(A, a_scale, a_zero_point, B, b_scale, b_zero_point, C, y_scale, y_zero_point, transB)

    A_info = helper.make_tensor_value_info("A", dtype_proto, list(A.shape))
    a_scale_info = helper.make_tensor_value_info("a_scale", onnx.TensorProto.FLOAT, [])
    a_zp_info = helper.make_tensor_value_info("a_zero_point", dtype_proto, [])
    B_info = helper.make_tensor_value_info("B", dtype_proto, list(B.shape))
    b_scale_info = helper.make_tensor_value_info("b_scale", onnx.TensorProto.FLOAT, [])
    b_zp_info = helper.make_tensor_value_info("b_zero_point", dtype_proto, [])
    C_info = helper.make_tensor_value_info("C", onnx.TensorProto.INT32, list(C.shape) if C.ndim > 0 else [])
    y_scale_info = helper.make_tensor_value_info("y_scale", onnx.TensorProto.FLOAT, [])
    y_zp_info = helper.make_tensor_value_info("y_zero_point", dtype_proto, [])
    Y_info = helper.make_tensor_value_info("Y", dtype_proto, list(Y.shape))

    attrs = {}
    if transB:
        attrs["transB"] = 1

    node = helper.make_node(
        "QGemm",
        ["A", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point", "C", "y_scale", "y_zero_point"],
        ["Y"],
        domain="com.microsoft",
        **attrs,
    )

    graph = helper.make_graph(
        [node],
        name,
        [A_info, a_scale_info, a_zp_info, B_info, b_scale_info, b_zp_info, C_info, y_scale_info, y_zp_info],
        [Y_info],
    )

    model = helper.make_model(
        graph,
        producer_name="qgemm.py",
        opset_imports=[helper.make_opsetid("com.microsoft", 1)],
    )

    # Some onnx versions may not have QGemm schema; continue even if checker fails.
    try:
        onnx.checker.check_model(model)
    except Exception:
        pass

    test_dir = os.path.join(BASE_DIR, f"test_{name}")
    data_dir = os.path.join(test_dir, "test_data_set_0")
    os.makedirs(data_dir, exist_ok=True)

    onnx.save(model, os.path.join(test_dir, "model.onnx"))

    inputs = [A, a_scale, a_zero_point, B, b_scale, b_zero_point, C, y_scale, y_zero_point]
    for i, t in enumerate(inputs):
        save_tensor(t, os.path.join(data_dir, f"input_{i}.pb"))
    save_tensor(Y, os.path.join(data_dir, "output_0.pb"))

    print(f"local_node_test({name})")


make_test(
    name="qgemm_uint8_scalar_bias",
    dtype_proto=onnx.TensorProto.UINT8,
    A=[[130, 132, 127], [129, 125, 133]],
    a_scale=0.02,
    a_zero_point=128,
    B=[[126, 131], [129, 127], [132, 124]],
    b_scale=0.03,
    b_zero_point=128,
    C=3,
    y_scale=0.015,
    y_zero_point=127,
)

make_test(
    name="qgemm_int8_bias_per_output",
    dtype_proto=onnx.TensorProto.INT8,
    A=[[-3, 2, 7], [4, -5, 1]],
    a_scale=0.05,
    a_zero_point=0,
    B=[[6, -2], [1, 3], [-4, 5]],
    b_scale=0.04,
    b_zero_point=0,
    C=[2, -3],
    y_scale=0.02,
    y_zero_point=-2,
)

make_test(
    name="qgemm_uint8_transB_row_bias",
    dtype_proto=onnx.TensorProto.UINT8,
    A=[[140, 120, 130], [150, 110, 100]],
    a_scale=0.01,
    a_zero_point=128,
    B=[[125, 130, 135], [132, 120, 110]],
    b_scale=0.02,
    b_zero_point=128,
    C=[[4], [-6]],
    y_scale=0.01,
    y_zero_point=120,
    transB=1,
)
