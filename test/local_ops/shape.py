# Generate a ONNX-style backend test
import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

test_name="test_shape_const_out"

A = np.random.rand( 5, 12 ).astype(np.float32)

axes = np.array( [1] )
starts = np.array( [1] )
ends = np.array( [4] )
steps = np.array( [2] )

g = so.empty_graph()
n1 = so.node('Shape', inputs=['data'], outputs=['Y'])
n2 = so.node('Expand', inputs=['data', 'Y'], outputs=['O'])

g = so.add_node(g, n1)
g = so.add_node(g, n2)
g = so.add_input(g, 'data', "FLOAT", A.shape)

g = so.add_output(g, 'O', "FLOAT", (5,12))


so.check(g)

example = {
	"data": A,
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["O"]
                )
print(result)


def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(A, test_name + "/test_data_set_0/input_0.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
