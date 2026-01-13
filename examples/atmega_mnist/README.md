Demo on running a hand-written number recognition (MNIST)
neural network on a Arduino Uno board (Microchip Atmega 8-bit MCU).

Onnx2c's (experimental) built-in quantization was removed in commit
`fe6226bac92a0a5cff5e42b8ebf8156e9e9e612a`, after which this example
does not build anymore. It might be fixed by using external quantization
tools in the future. The example is left intact as a reference.


Usage:

	make
	cd avr
	make

No checks for prerequisites are made. See the Makefiles for details.

The network is trained from scratch, so the first step in the Makefile takes
a good while. A trained .onnx network file is provided.
