# Generate the local Gemm regression tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

test_name="test_resize_downsample_sizes_linear_2D_align"

X = np.array([[[
 [1,2,3,4],
 [5,6,7,8]]]],
# [11,12,13,14,15],
# [16,17,18,19,20],
# [21,22,23,24,25]]
 ).astype(np.float32)
#X = np.array( [[1,2,3,4,5]] ).astype(np.float32)

# NB: when using sizes, roi and scales MUST be given to so.node(), but
# they also MUST be set to empty tensors for onnxruntime to run inference.
# However, when producing onnx2c backend tests roi and scales MUST NOT be given.
# This however creates graphs that cause onnxruntime to choke :|
roi = np.array([]).astype(np.int64)
scales = np.array([1,1,0.6,0.6]).astype(np.float32)
sizes = np.array([1,1,1,2]).astype(np.int64)

g = so.empty_graph()

n1 = so.node('Resize', inputs=['X', 'roi', 'scales', ''], mode='linear', coordinate_transformation_mode='align_corners',outputs=['Y'])
#n1 = so.node('Resize', inputs=['X', '', 'scales', ''], mode='linear', coordinate_transformation_mode='align_corners',outputs=['Y'])
g = so.add_node(g, n1)
g = so.add_input(g, 'X', "FLOAT", X.shape)
g = so.add_constant(g, 'roi', roi, "FLOAT")
g = so.add_constant(g, 'scales', scales, "FLOAT")
#g = so.add_constant(g, 'sizes', sizes, "INT64")

g = so.add_output(g, 'Y', "FLOAT", (1,4))


so.check(g)

example = {
	"X": X,
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["Y"]
                )
print("input:")
print(X)
print("output:")
print(result)


def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(X, test_name + "/test_data_set_0/input_0.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
