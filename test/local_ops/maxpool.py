# Generate a onnx2c local tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

test_name="test_maxpool_stride_1"

X = np.random.randint(-100, 100, (1, 3, 12, 12)).astype(np.float32)
X = X / 100
data = np.array([X,X,X,X])

g = so.empty_graph()

n1 = so.node('MaxPool', inputs=['X'],outputs=['Y', 'indices'], kernel_shape=[1,1], strides=[1,1])
g = so.add_node(g, n1)
g = so.add_input(g, 'X', "FLOAT", X.shape)
g = so.add_output(g, 'Y', "FLOAT", [1,3,12,12])
g = so.add_output(g, 'indices', "INT64", [1,3,12,12])


so.check(g)

example = {
	"X": X,
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["Y", "indices"]
                )

def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(X, test_name + "/test_data_set_0/input_0.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
save_tensor(result[1], test_name + "/test_data_set_0/output_1.pb")
