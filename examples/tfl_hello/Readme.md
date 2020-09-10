"Hello world"
============

This is a fancy embedded "Hello World" application. It

 - uses a neural network to compute sin(x)
 - uses sin(x) to drive a PWM signal to blink a LED


This is the "Hello World" example mentioned in the
onnx2c top level Reamde.



Building
--------

You will need

 - onnx2c built & found your `PATH`
 - STM32F411-Discovery board
 - LibOpemCM3 built and found in `OPENCM3_DIR`
 - GCC cross compiler
 - OpenOCD

Build by

	make
	make flash

Please see the Makefile for details.

