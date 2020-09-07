# Trivial LSTM generator
# This is intended to generate onnx models
# used as tests in onnx2c.
# NB: this script does not train the model
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import keras2onnx


if __name__ == "__main__":

	input_layer = keras.layers.Input(shape=(None,5))
	x = keras.layers.LSTM(3,
		kernel_initializer="glorot_uniform",
		#kernel_initializer="ones",
		#kernel_initializer="identity",
		#recurrent_initializer="orthogonal",
		#bias_initializer="zeros",
		bias_initializer="glorot_uniform",
		#bias_initializer="ones",
		unit_forget_bias=False,
		#activation="relu",
		#recurrent_activation="relu",
		)(input_layer)
	
	model = keras.Model(input_layer, x) 
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
	model.compile(optimizer=optimizer, loss="MSE")

	num_rounds=5
	inp = keras.utils.to_categorical(np.zeros(num_rounds), num_classes=5)
	inp = inp[np.newaxis, ...]

	print(inp)
	predictions = model.predict(inp)[0]

	print(predictions)

	onnx_model = keras2onnx.convert_keras(model, "lstm")
	keras2onnx.save_model(onnx_model, "lstm.onnx")
