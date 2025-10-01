# Generates a onnx-style backend test
#

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import torch
import onnx
import numpy as np
from onnx import numpy_helper, TensorProto

def save_to_onnx_test_case(model: torch.nn.Module,
                           input_tensors: list[torch.Tensor],
                           output_tensors: list[torch.Tensor],
                           root_dir: str = None,
                           dataset_no: int = 0,
                           opset_version: int = 13,
                           ):
    """
    Save PyTorch model, example input, and expected output in ONNX backend unit test style.

    """

    input_tuple = tuple(input_tensors)
    output_tuple = tuple(output_tensors)
    input_names = [f"input_{i}" for i in range(len(input_tuple))]
    output_names = [f"output_{i}" for i in range(len(output_tuple))]

    # Create directory structure
    if root_dir is None:
        root_dir = os.getcwd()
    os.makedirs(root_dir, exist_ok=True)
    model_path = os.path.join(root_dir, "model.onnx")
    test_data_dir = os.path.join(root_dir, f"test_data_set_{dataset_no}")
    os.makedirs(test_data_dir, exist_ok=True)

    # Export model to ONNX
    # We need to give example inputs, input_names, output_names
    torch.onnx.export(model,
                      input_tuple,
                      model_path,
                      export_params=True,
                      opset_version=opset_version,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names)

    # Function to serialize a numpy array (or pytorch tensor) as .pb file (TensorProto)
    def save_tensor_as_pb(tensor: np.ndarray, filename: str):
        tensor = np.asarray(tensor)
        tensor_proto = numpy_helper.from_array(tensor)
        # write to file
        with open(filename, "wb") as f:
            f.write(tensor_proto.SerializeToString())

    # Save input(s)
    for idx, inp in enumerate(input_tuple):
        inp_np = inp.cpu().numpy()
        fname = os.path.join(test_data_dir, f"input_{idx}.pb")
        save_tensor_as_pb(inp_np, fname)

    # Save output(s)
    for idx, out in enumerate(output_tuple):
        out_np = out.cpu().numpy()
        fname = os.path.join(test_data_dir, f"output_{idx}.pb")
        save_tensor_as_pb(out_np, fname)

    print(f"Saved ONNX model to {model_path}")
    print(f"Saved test data to {test_data_dir}")


class ModelWithConstParam(nn.Module):
    def __init__(self):
        super(ModelWithConstParam, self).__init__()
        # Create a single layer network, where the layer is a multiplication, and
        # takes a constant parameter
        

    def forward(self, x):
        x = x * 2.0
        return x

def create_onnx_function_model():
    model = ModelWithConstParam()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 10)

    save_to_onnx_test_case(
        model,
        [dummy_input],
        [model(dummy_input)],
        root_dir=None,
        dataset_no=0,
        opset_version=13
    )

    print("Model has been exported to functions.onnx")

if __name__ == "__main__":
    create_onnx_function_model()

