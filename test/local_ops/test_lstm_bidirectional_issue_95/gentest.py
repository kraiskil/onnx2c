# Generates ONNX style test for bidirectional LSTM.
# AI generated

import os
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

def build_simple_lstm_model(seq_length: int,
                            batch_size: int,
                            input_size: int,
                            hidden_size: int,
                            num_directions: int = 1,
                            opset: int = 18) -> onnx.ModelProto:
    """
    Build a minimal ONNX model with a single LSTM node.
    """
    # Random weights, bias, and initial states
    W = np.random.randn(num_directions, 4 * hidden_size, input_size).astype(np.float32)
    R = np.random.randn(num_directions, 4 * hidden_size, hidden_size).astype(np.float32)
    B = np.random.randn(num_directions, 8 * hidden_size).astype(np.float32)
    initial_h = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)
    initial_c = np.zeros((num_directions, batch_size, hidden_size), dtype=np.float32)

    # Define input tensor
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT,
                                      [seq_length, batch_size, input_size])

    # Define outputs
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT,
                                      [seq_length, num_directions, batch_size, hidden_size])
    Y_h = helper.make_tensor_value_info("Y_h", TensorProto.FLOAT,
                                        [num_directions, batch_size, hidden_size])
    Y_c = helper.make_tensor_value_info("Y_c", TensorProto.FLOAT,
                                        [num_directions, batch_size, hidden_size])

    # Create initial states and weights as initializers (constants)
    W_init = numpy_helper.from_array(W, name="W")
    R_init = numpy_helper.from_array(R, name="R")
    B_init = numpy_helper.from_array(B, name="B")
    h0_init = numpy_helper.from_array(initial_h, name="initial_h")
    c0_init = numpy_helper.from_array(initial_c, name="initial_c")

    # Build the LSTM node
    lstm_node = helper.make_node(
        "LSTM",
        inputs=["X", "W", "R", "B", "", "initial_h", "initial_c"],
        outputs=["Y", "Y_h", "Y_c"],
        hidden_size=hidden_size,
        direction="bidirectional"
    )

    graph = helper.make_graph(
        nodes=[lstm_node],
        name="SimpleLSTMGraph",
        inputs=[X],
        outputs=[Y, Y_h, Y_c],
        initializer=[W_init, R_init, B_init, h0_init, c0_init]
    )

    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid("", opset)])
    model.ir_version = 11
    return model

def save_tensor_as_pb(tensor: np.ndarray, path: str):
    """
    Save a NumPy array as a TensorProto .pb file.
    """
    arr = np.ascontiguousarray(tensor)
    tensor_proto = numpy_helper.from_array(arr)
    with open(path, "wb") as f:
        f.write(tensor_proto.SerializeToString())

def generate_and_save_test_case(root_dir: str,
                                model: onnx.ModelProto,
                                input_data: np.ndarray):
    """
    Given an ONNX model and an input sample, compute the outputs via onnxruntime,
    and save the model plus input/output in ONNX-backend-test style.
    """
    import onnxruntime as ort

    os.makedirs(root_dir, exist_ok=True)
    model_path = os.path.join(root_dir, "model.onnx")
    onnx.save(model, model_path)

    # Use ONNX Runtime to run inference
    sess = ort.InferenceSession(model_path)
    # The model expects a single input "X"
    outputs = sess.run(None, {"X": input_data})

    # Prepare test_data_set directory
    test_data_dir = os.path.join(root_dir, "test_data_set_0")
    os.makedirs(test_data_dir, exist_ok=True)

    # Save input(s)
    save_tensor_as_pb(input_data, os.path.join(test_data_dir, "input_0.pb"))

    # Save output(s)
    for idx, out in enumerate(outputs):
        save_tensor_as_pb(out, os.path.join(test_data_dir, f"output_{idx}.pb"))

    print(f"Saved ONNX model: {model_path}")
    print(f"Saved test data under: {test_data_dir}")

def main():
    # Define dimensions
    seq_length = 5
    batch_size = 1
    input_size = 4
    hidden_size = 3
    num_directions = 2

    model = build_simple_lstm_model(seq_length,
                                    batch_size,
                                    input_size,
                                    hidden_size,
                                    num_directions,
                                    opset=18)

    # Create a random input sample
    input_sample = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)

    # Directory to store model + test data
    out_dir = "."
    generate_and_save_test_case(out_dir, model, input_sample)

if __name__ == "__main__":
    main()

