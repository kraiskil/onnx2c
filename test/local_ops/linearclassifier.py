# Generate the local ai.onnx.ml LinearClassifier regression tests.
# The expected outputs in the test_data_set directories are
# computed with onnxruntime.

import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

np.random.seed(42)


def save_tensor(t, fn):
	with open(fn, "wb") as f:
		f.write(numpy_helper.from_array(t).SerializeToString())


def make_test(name, coefficients, intercepts, classlabels_ints,
              post_transform, multi_class, num_examples, num_features):
	coefficients = np.asarray(coefficients, dtype=np.float32)
	intercepts = np.asarray(intercepts, dtype=np.float32)
	classlabels_ints = [int(c) for c in classlabels_ints]
	num_classes = len(intercepts)
	# binary (single intercept) scores are expanded to 2 columns
	score_classes = 2 if num_classes == 1 else num_classes

	assert coefficients.size == num_classes * num_features

	X = np.random.rand(num_examples, num_features).astype(np.float32)

	attrs = {
		"coefficients": coefficients.tolist(),
		"intercepts": intercepts.tolist(),
		"classlabels_ints": classlabels_ints,
		"post_transform": post_transform,
		"multi_class": multi_class,
	}
	node = helper.make_node(
		"LinearClassifier", ["X"], ["Y", "Z"],
		domain="ai.onnx.ml", name=name, **attrs)

	input_vi = helper.make_tensor_value_info(
		"X", onnx.TensorProto.FLOAT, [num_examples, num_features])
	label_vi = helper.make_tensor_value_info(
		"Y", onnx.TensorProto.INT64, [num_examples])
	scores_vi = helper.make_tensor_value_info(
		"Z", onnx.TensorProto.FLOAT, [num_examples, score_classes])

	graph = helper.make_graph(
		[node], name, [input_vi], [label_vi, scores_vi])
	model = helper.make_model(
		graph, producer_name="linearclassifier.py",
		opset_imports=[helper.make_opsetid("", 13),
		              helper.make_opsetid("ai.onnx.ml", 1)])
	# IR version 14 (onnx >= 1.18) is not yet supported by onnxruntime
	model.ir_version = 8
	onnx.checker.check_model(model)

	dir_name = "test_" + name
	os.makedirs(os.path.join(dir_name, "test_data_set_0"), exist_ok=True)
	onnx.save(model, os.path.join(dir_name, "model.onnx"))

	sess = ort.InferenceSession(model.SerializeToString())
	labels, scores = sess.run(None, {"X": X})

	save_tensor(X, os.path.join(dir_name, "test_data_set_0", "input_0.pb"))
	save_tensor(labels, os.path.join(dir_name, "test_data_set_0", "output_0.pb"))
	save_tensor(scores, os.path.join(dir_name, "test_data_set_0", "output_1.pb"))


# Binary classification, integer labels
make_test(
	name="linearclassifier_binary_int",
	coefficients=[0.5, -1.25, 0.75],
	intercepts=[0.4],
	classlabels_ints=[7, 3],
	post_transform="NONE",
	multi_class=0,
	num_examples=4,
	num_features=3)

# Binary classification with a logistic score transformation
make_test(
	name="linearclassifier_binary_logistic",
	coefficients=[-0.5, 0.25, 1.5],
	intercepts=[-0.25],
	classlabels_ints=[7, 3],
	post_transform="LOGISTIC",
	multi_class=0,
	num_examples=4,
	num_features=3)

# Multiclass classification
make_test(
	name="linearclassifier_multiclass",
	coefficients=[0.5, -0.25, 0.75, -1.0, 0.2, 0.1, 0.3, 0.9, -0.4],
	intercepts=[0.0, 0.0, 0.0],
	classlabels_ints=[2, 5, 11],
	post_transform="NONE",
	multi_class=0,
	num_examples=4,
	num_features=3)

# Multiclass with a logistic score transformation
make_test(
	name="linearclassifier_multiclass_logistic",
	coefficients=[0.5, -0.25, 0.75, -1.0, 0.2, 0.1, 0.3, 0.9, -0.4],
	intercepts=[0.3, -0.2, 0.1],
	classlabels_ints=[4, 8, 15],
	post_transform="LOGISTIC",
	multi_class=0,
	num_examples=4,
	num_features=3)

# Multiclass with non-zero intercepts and a softmax score transformation
make_test(
	name="linearclassifier_multiclass_softmax",
	coefficients=[0.5, -0.25, 0.75, -1.0, 0.2, 0.1, 0.3, 0.9, -0.4],
	intercepts=[0.3, -0.2, 0.1],
	classlabels_ints=[0, 1, 2],
	post_transform="SOFTMAX",
	multi_class=1,
	num_examples=4,
	num_features=3)

# Multiclass with a softmax-zero score transformation
make_test(
	name="linearclassifier_multiclass_softmax_zero",
	coefficients=[0.5, -0.25, 0.75, -1.0, 0.2, 0.1, 0.3, 0.9, -0.4],
	intercepts=[0.3, -0.2, 0.1],
	classlabels_ints=[0, 1, 2],
	post_transform="SOFTMAX_ZERO",
	multi_class=0,
	num_examples=4,
	num_features=3)
