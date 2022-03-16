# Generate a onnx2c local tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

test_name="test_scatternd_indices_1x1x2"

X = np.array([
 [1,2,3,4],
 [5,6,7,8],
 [8,7,6,5],
 [4,3,2,1]
]
).astype(np.float32)
# 4x4x4
data = np.array([X,X,X,X])

# Test case indices_3x2
#indices = np.array(
#  [
#   [0,0],
#   [1,1],
#   [2,3],
#  ]
#).astype(np.int64)
#updates = np.array([
#  [0,0,1,0],
#  [42,42,0,42],
#  [0x42,0,0,0x42],
#]).astype(np.float32)

# Test case indices_3x3
indices = np.array(
  [
   [[0,2]],
  ]
).astype(np.int64)
updates = np.array([
  [[0,0,42,0]],
  #[[42,42,0,42], [42,42,0,42]]
]).astype(np.float32)




print("data shape: ", data.shape)
print("indi shape: ", indices.shape)
print("upda shape: ", updates.shape)
print("expected update shape (per spec): ", indices.shape[-1],  data.shape[indices.shape[-1]:] )
print("expected update shape (per impl): ", indices.shape[0],  data.shape[indices.shape[-1]:] )

g = so.empty_graph()

n1 = so.node('ScatterND', inputs=['data', 'indices', 'updates'],outputs=['output'])
g = so.add_node(g, n1)
g = so.add_input(g, 'data', "FLOAT", data.shape)
g = so.add_input(g, 'indices', "INT64", indices.shape)
g = so.add_input(g, 'updates', "FLOAT", updates.shape)

g = so.add_output(g, 'output', "FLOAT", data.shape)


so.check(g)

example = {
	"data": data,
	"indices": indices,
	"updates": updates
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["output"]
                )
print("input:")
print(X)
print("output:")
print(result)


def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(data, test_name + "/test_data_set_0/input_0.pb")
save_tensor(indices, test_name + "/test_data_set_0/input_1.pb")
save_tensor(updates, test_name + "/test_data_set_0/input_2.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
