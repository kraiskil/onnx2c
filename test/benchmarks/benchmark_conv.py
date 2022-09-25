# Generate a onnx2c local tests
# Each run generates one test, alter test_name and variable
# between runs.

import numpy as np
import sclblonnx as so
from onnx import helper, numpy_helper
from pathlib import Path


# Values for yolov6n first convolution
#test_name="benchmark_conv_yolov6n_inputlayer"
#maps = 16
#channels = 3
#strides = [2,2]
#dilations = [1,1]
#kernel_size = [3,3]
#pads=[1,1,1,1]
#in_size = [1, channels, 640, 640]
#out_size = [1, maps, 320, 320]
#w_size = [maps, channels] + kernel_size

# Values for yolov6n conv4, slowest to compute
test_name="benchmark_conv_fits_128k"
maps = 28
channels = 28
strides = [1,1]
dilations = [1,1]
kernel_size = [3,3]
pads=[1,1,1,1]
in_size = [1, channels, 20, 20]
out_size = [1, maps, 20, 20]
w_size = [maps, channels] + kernel_size


# Values for yolov6n last convolution
#test_name="benchmark_conv_yolov6n_lastconv"
#maps = 1
#channels = 128
#strides = [1,1]
#dilations = [1,1]
#kernel_size = [1,1]
#pads=[0,0,0,0]
#in_size = [1, channels, 20, 20]
#out_size = [1, maps, 20, 20]
#w_size = [maps, channels] + kernel_size



X = np.random.random(in_size).astype(np.float32)
w = np.random.random(w_size).astype(np.float32)

g = so.empty_graph()

n1 = so.node('Conv', inputs=['X', 'w'],outputs=['Y'], kernel_shape=kernel_size, strides=strides, dilations=dilations, pads=pads)
g = so.add_node(g, n1)
g = so.add_input(g, 'X', "FLOAT", X.shape)
g = so.add_input(g, 'w', "FLOAT", w.shape)
g = so.add_output(g, 'Y', "FLOAT", out_size)


so.check(g)

example = {
	"X": X,
	"w": w,
}
Path(test_name + "/test_data_set_0").mkdir(parents=True, exist_ok=True)
so.graph_to_file(g, test_name + "/model.onnx")
result = so.run(g,
                inputs=example,
                outputs=["Y"]
                )

def save_tensor(t, fn):
	with open(fn, 'wb') as f:
		npt = numpy_helper.from_array(t)
		f.write(npt.SerializeToString(npt))

save_tensor(X, test_name + "/test_data_set_0/input_0.pb")
save_tensor(w, test_name + "/test_data_set_0/input_1.pb")
save_tensor(result[0], test_name + "/test_data_set_0/output_0.pb")
