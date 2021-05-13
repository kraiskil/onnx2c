Demo on running a hand-written number recognition (MNIST)
neural network on a Arduino Uno board (Microchip Atmega 8-bit MCU).

Usage:

	make
	cd avr
	make

No checks for prerequisites are made. See the Makefiles for details.

The network is trained from scratch, so the first step in the Makefile takes
a good while. A trained .onnx network file is provided.
