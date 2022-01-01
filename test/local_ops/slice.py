# Generate the local Gemm regression tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

test_name="test_slice_end_INT64_MAX"

A = np.random.rand( 5, 12 ).astype(np.float32)

axes = np.array( [1] )
starts = np.array( [1] )
ends = np.array( [9223372036854775807] )
steps = np.array( [2] )

g = so.empty_graph()
n1 = so.node('Slice', inputs=['data', 'starts', 'ends', 'axes', 'steps'], outputs=['Y'])
g = so.add_node(g, n1)
g = so.add_input(g, 'data', "FLOAT", A.shape)
g = so.add_constant(g, 'starts', starts, "INT64")
g = so.add_constant(g, 'ends',   ends,   "INT64")
g = so.add_constant(g, 'axes',   axes,   "INT64")
g = so.add_constant(g, 'steps',  steps,  "INT64")

g = so.add_output(g, 'Y', "FLOAT", (5,6))


so.check(g)

example = {
	"data": A,
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["Y"]
                )
print(result)


def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(A, test_name + "/test_data_set_0/input_0.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
