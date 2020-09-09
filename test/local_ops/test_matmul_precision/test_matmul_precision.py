import onnx
from onnx import helper
from onnx import TensorProto
import numpy
from onnx import numpy_helper
import onnxruntime as rt

# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[10.10, 11.11], [12.12, 13.13]], dtype=numpy.float32)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# Save the TensorProto
with open('test_data_set_0/input_0.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

# Save the TensorProto again
with open('test_data_set_0/input_1.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

################################

# Create one input (ValueInfoProto)
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 2])

# Create one output (ValueInfoProto)
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 2])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'MatMul',    # node name
    ['A', 'B'],  # inputs
    ['C'],       # outputs,
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    'test-model',
    [A, B],
    [C],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

onnx.save(model_def, 'model.onnx')

##################################################

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')

##################################################

# create runtime session
sess = rt.InferenceSession("model.onnx")

print("-------------\n")
# get output name
input_name1 = sess.get_inputs()[0].name
print("input name", input_name1)
input_name2 = sess.get_inputs()[1].name
print("input name", input_name2)
output_name = sess.get_outputs()[0].name
print("output name", output_name)
output_shape = sess.get_outputs()[0].shape
print("output shape", output_shape)
print("-------------\n")

# forward model
res = sess.run([output_name], {input_name1: numpy_array, input_name2: numpy_array})
out = numpy.array(res)
print(out)
print("-------------\n")

print('Original Numpy array:\n{}\n'.format(out[0]))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(out[0])
print('TensorProto:\n{}'.format(tensor))

# Save the TensorProto
with open('test_data_set_0/output_0.pb', 'wb') as f:
    f.write(tensor.SerializeToString())
print("-------------\n")
