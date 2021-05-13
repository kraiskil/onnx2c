#
# Makefile for host-side testing utilities
# For the AVR makefile, see the avr/ directory
#

all: test_float test_int avr/generated_avr.c
.PHONY: all
.PRECIOUS: mnist.onnx
.INTERMEDIATE: mnist.simplified.onnx

OPT=-O0 -g

# Train the neural network. Needs pytorch
mnist.onnx:
	python3 create_network.py

# Simplify pythorch output. Needs https://github.com/daquexian/onnx-simplifier
# (why does pytorch make unnecessary complex networks?)
mnist.simplified.onnx: mnist.onnx
	python3 -m onnxsim mnist.onnx mnist.simplified.onnx

# Generate C out of the network.
# The three versions are:
#  - using floating point arithmetic
#  - using quantized 8bit integer arithmetic
#  - 8bit integer arithmetic, with AVR specifics to store consts in instruction memory
generated_float.c: mnist.simplified.onnx
	../../onnx2c mnist.simplified.onnx > generated_float.c
generated_int.c: mnist.simplified.onnx
	../../onnx2c -quantize mnist.simplified.onnx > generated_int.c
avr/generated_avr.c: mnist.simplified.onnx
	../../onnx2c -quantize -avr mnist.simplified.onnx > avr/generated_avr.c


# Create and run the host-side "unit tests"
test_float: test_float.c load.c generated_float.c
	gcc $^ -o $@ -Wall $(OPT)
test_int: test_int.c load.c generated_int.c
	gcc $^ -o $@ -Wall $(OPT) -Wno-unused-variable

test: test_float test_int
	@echo float
	./test_float
	@echo  quantized
	./test_int
