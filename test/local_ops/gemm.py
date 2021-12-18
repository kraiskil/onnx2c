# Generate the local Gemm regression tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path

# gemm computes Y=a*A*B+b*C where
# dim(A) = M,K
# dim(B) = K,N
# either or both A and B can be given transposed
M=2
K=3
N=4
a=1.0
b=3.2
transA=1
transB=1
C = np.random.rand(N).astype(np.float32)
test_name="test_gemm_CN_transA_transB"

if transA:
	A = np.random.rand( K, M ).astype(np.float32)
else:
	A = np.random.rand( M, K ).astype(np.float32)
if transB:
	B = np.random.rand( N, K).astype(np.float32)
else:
	B = np.random.rand( K, N).astype(np.float32)

print(A.shape)
print(B.shape)

g = so.empty_graph()
# B and sequence_lengths missing
n1 = so.node('Gemm', inputs=['A', 'B', 'C'], outputs=['Y'], transA=transA, transB=transB, alpha=a, beta=b)
g = so.add_node(g, n1)
g = so.add_input(g, 'A', "FLOAT", A.shape)
g = so.add_input(g, 'B', "FLOAT", B.shape)
g = so.add_input(g, 'C', "FLOAT", C.shape)
g = so.add_output(g, 'Y', "FLOAT", (M,N))


so.check(g)

example = {
	"A": A,
	"B": B,
	"C": C
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
save_tensor(B, test_name + "/test_data_set_0/input_1.pb")
save_tensor(C, test_name + "/test_data_set_0/input_2.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
