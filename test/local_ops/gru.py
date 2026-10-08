# Generate the local GRU regression tests (all test_gru_* directories).
# Expected outputs are computed with onnxruntime, except layout=1 (no ORT
# kernel for that) which uses a transposed layout=0 twin model instead.
# Run with an onnx/onnxruntime-capable python3.
import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper
import onnxruntime as ort

BASE = os.path.dirname(os.path.abspath(__file__))


def build(name, seq, batch, ds, hs, n_dir, direction, layout, lbr, with_B, with_initial_h,
          outputs, seed=42):
    rng = np.random.default_rng(seed)
    x_shape_in = (batch, seq, ds) if layout == 1 else (seq, batch, ds)
    X = rng.standard_normal(x_shape_in).astype(np.float32) * 0.5
    W = rng.standard_normal((n_dir, 3 * hs, ds)).astype(np.float32) * 0.3
    R = rng.standard_normal((n_dir, 3 * hs, hs)).astype(np.float32) * 0.3
    B = rng.standard_normal((n_dir, 6 * hs)).astype(np.float32) * 0.1 if with_B else None
    # initial_h: layout 0 -> [dir, batch, hs]; layout 1 -> [batch, dir, hs]
    if with_initial_h:
        H = rng.standard_normal(
            (n_dir, batch, hs) if layout == 0 else (batch, n_dir, hs)
        ).astype(np.float32) * 0.5
    else:
        H = None

    if layout == 0:
        x_shape, y_shape = [seq, batch, ds], [seq, n_dir, batch, hs]
        h_shape = (n_dir, batch, hs) if with_initial_h else None
    else:
        x_shape, y_shape = [batch, seq, ds], [batch, seq, n_dir, hs]
        h_shape = (batch, n_dir, hs) if with_initial_h else None

    node_inputs = ["X", "W", "R"]
    feed = [X, W, R]
    if with_B:
        node_inputs.append("B")
        feed.append(B)
    if with_initial_h:
        node_inputs.append("")  # sequence_lens omitted
        node_inputs.append("initial_h")
        feed.append(H)
    elif with_B:
        pass  # trailing optionals omitted: fine
    # note: if initial_h without B: ["X","W","R","","","initial_h"] - not used here

    node = helper.make_node(
        "GRU", node_inputs, outputs,
        hidden_size=hs, direction=direction, layout=layout,
        linear_before_reset=lbr,
    )

    graph_inputs = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, x_shape),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, [n_dir, 3 * hs, ds]),
        helper.make_tensor_value_info("R", TensorProto.FLOAT, [n_dir, 3 * hs, hs]),
    ]
    if with_B:
        graph_inputs.append(
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [n_dir, 6 * hs]))
    if with_initial_h:
        graph_inputs.append(
            helper.make_tensor_value_info("initial_h", TensorProto.FLOAT, list(h_shape)))

    graph_outputs = []
    if "Y" in outputs:
        graph_outputs.append(
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, y_shape))
    if "Y_h" in outputs:
        yh_shape = list(h_shape) if with_initial_h else (
            (n_dir, batch, hs) if layout == 0 else (batch, n_dir, hs))
        graph_outputs.append(
            helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, yh_shape))

    graph = helper.make_graph([node], "gru_local", graph_inputs, graph_outputs)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)],
                              ir_version=8)
    onnx.checker.check_model(model)

    ort_in = {"X": X, "W": W, "R": R}
    if with_B:
        ort_in["B"] = B
    if with_initial_h:
        ort_in["initial_h"] = H
    try:
        sess = ort.InferenceSession(model.SerializeToString(),
                                    providers=["CPUExecutionProvider"])
        results = sess.run(None, ort_in)
        via = "ort"
    except Exception:
        # ORT's CPU EP has no layout=1 GRU kernel. Compute the expected
        # outputs from an equivalent layout=0 twin model (ORT) instead,
        # transposing X/initial_h in and Y/Y_h out.
        X0 = np.transpose(X, (1, 0, 2))
        feeds0 = [X0, W, R] + ([B] if with_B else [])
        if with_initial_h:
            feeds0.append(np.transpose(H, (1, 0, 2)))
        node0 = helper.make_node(
            "GRU", ["" if x == "" else x for x in node_inputs], outputs,
            hidden_size=hs, direction=direction, layout=0,
            linear_before_reset=lbr)
        gin0 = [
            helper.make_tensor_value_info("X", TensorProto.FLOAT, [seq, batch, ds]),
            helper.make_tensor_value_info("W", TensorProto.FLOAT, [n_dir, 3 * hs, ds]),
            helper.make_tensor_value_info("R", TensorProto.FLOAT, [n_dir, 3 * hs, hs]),
        ]
        if with_B:
            gin0.append(helper.make_tensor_value_info(
                "B", TensorProto.FLOAT, [n_dir, 6 * hs]))
        if with_initial_h:
            gin0.append(helper.make_tensor_value_info(
                "initial_h", TensorProto.FLOAT, [n_dir, batch, hs]))
        gout0 = []
        if "Y" in outputs:
            gout0.append(helper.make_tensor_value_info(
                "Y", TensorProto.FLOAT, [seq, n_dir, batch, hs]))
        if "Y_h" in outputs:
            gout0.append(helper.make_tensor_value_info(
                "Y_h", TensorProto.FLOAT, [n_dir, batch, hs]))
        graph0 = helper.make_graph([node0], "g0", gin0, gout0)
        model0 = helper.make_model(graph0, opset_imports=[helper.make_opsetid("", 17)],
                                   ir_version=8)
        names0 = [i.name for i in gin0]
        feed0 = dict(zip(names0, feeds0))
        sess0 = ort.InferenceSession(model0.SerializeToString(),
                                     providers=["CPUExecutionProvider"])
        res0 = sess0.run(None, feed0)
        results = []
        for oname, r in zip(outputs, res0):
            if oname == "Y":        # [seq, dir, batch, hs] -> [batch, seq, dir, hs]
                results.append(np.transpose(r, (2, 0, 1, 3)))
            else:                   # Y_h: [dir, batch, hs] -> [batch, dir, hs]
                results.append(np.transpose(r, (1, 0, 2)))
        via = "ort-layout0-twin"
    print(f"{name}: expected outputs via {via}")

    d = os.path.join(BASE, "test_" + name)
    os.makedirs(os.path.join(d, "test_data_set_0"), exist_ok=True)
    onnx.save(model, os.path.join(d, "model.onnx"))

    for i, arr in enumerate(feed):
        with open(os.path.join(d, "test_data_set_0", f"input_{i}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(arr).SerializeToString())
    for i, arr in enumerate(results):
        with open(os.path.join(d, "test_data_set_0", f"output_{i}.pb"), "wb") as f:
            f.write(numpy_helper.from_array(arr).SerializeToString())
    print(f"{name}: OK ({len(feed)} inputs, {len(results)} outputs)")


build("gru_reverse", seq=4, batch=2, ds=3, hs=2, n_dir=1,
      direction="reverse", layout=0, lbr=0, with_B=True, with_initial_h=False,
      outputs=["Y", "Y_h"])

build("gru_bidirectional", seq=3, batch=2, ds=2, hs=3, n_dir=2,
      direction="bidirectional", layout=0, lbr=0, with_B=True, with_initial_h=False,
      outputs=["Y", "Y_h"])

# The reported use case: per-timestep GRU with explicit hidden state cache
build("gru_lbr1_initial_h", seq=3, batch=2, ds=3, hs=2, n_dir=1,
      direction="forward", layout=0, lbr=1, with_B=True, with_initial_h=True,
      outputs=["Y", "Y_h"])

build("gru_layout1_initial_h", seq=3, batch=2, ds=3, hs=2, n_dir=1,
      direction="forward", layout=1, lbr=0, with_B=True, with_initial_h=True,
      outputs=["Y", "Y_h"])

# Only the Y output is connected, no bias, no initial_h (memset path)
build("gru_y_only", seq=3, batch=2, ds=2, hs=2, n_dir=1,
      direction="forward", layout=0, lbr=0, with_B=False, with_initial_h=False,
      outputs=["Y"], seed=7)
