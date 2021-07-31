import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper


hidden_s=1
input_s=1
seq_len=1
batch_s=1
num_dir=1

x = np.array([1], np.float32).reshape([seq_len, batch_s, input_s])
w = np.array([1,1,1,1], np.float32).reshape([num_dir, 4*hidden_s, input_s])
r = np.array([1,1,1,1], np.float32).reshape([num_dir, 4*hidden_s, hidden_s])
init_h = np.array([4], np.float32).reshape([num_dir, batch_s, hidden_s])

print(x.shape)
print(w.shape)
print(r.shape)

g = so.empty_graph()
# B and sequence_lengths missing
n1 = so.node('LSTM', inputs=['x', 'w', 'r','', '', 'i_h'], outputs=['Y'], hidden_size=hidden_s)
g = so.add_node(g, n1)
g = so.add_input(g, 'x', "FLOAT", x.shape)
g = so.add_input(g, 'w', "FLOAT", w.shape)
g = so.add_input(g, 'r', "FLOAT", r.shape)
g = so.add_input(g, 'i_h', "FLOAT", init_h.shape)
g = so.add_output(g, 'Y', "FLOAT", [seq_len, num_dir, batch_s, hidden_s])


so.check(g)

example = {
	"x": x,
	"w": w,
	"r": r,
	"i_h": init_h
}
so.graph_to_file(g, "model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["Y"]
                )
print(result)


def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(x, "test_data_set_0/input_0.pb")
save_tensor(w, "test_data_set_0/input_1.pb")
save_tensor(r, "test_data_set_0/input_2.pb")
save_tensor(init_h, "test_data_set_0/input_3.pb")
save_tensor(result[0], "test_data_set_0/output_0.pb")
