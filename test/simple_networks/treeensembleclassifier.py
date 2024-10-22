from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
num_features = 25

# Train
clf = RandomForestClassifier().fit(X_train, y_train)

# Convert to ONNX
initial_type = [('float_input', FloatTensorType([1, num_features]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type, options={type(clf): {'zipmap': False}}, target_opset=12)

# Save
onnx_model_path = "treeensembleclassifier.onnx"
onnx.save(onnx_model, onnx_model_path)

# Write results to console
test_input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
results = ort.InferenceSession(onnx_model_path).run(None, {"float_input": np.array(test_input, dtype=np.float32).reshape(1, num_features)})

print(f"{results[0][0]}, {results[1][0][0]}, {results[1][0][1]}, {results[1][0][2]}")
