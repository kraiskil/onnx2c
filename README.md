onnx2c
======

An ONNX to C ~~compiler~~translator. This will read an ONNX file,
and generate C code to be included in your project.

Onnx2c's target is "edge computing", i.e. microcontrollers.

- generated code uses no printf or dynamic memory allocation


Currently onnx2c is a hobby project for me to learn ONNX,
edge computing and ML in general. Because of this, stuff of
little or no interest in onnx2c are:

 - ONNX specification coverage
 - accelerators


Building
--------

Make sure you have ProtocolBuffers libraries installed.

`mkdir build; cd build; cmake ..; make`


Onnx2c sources include generated protobuffer files. If you need to update them:

- get latest onnx.proto from ONXX github (https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)
- remove the last lines (i.e. option `optimize_for = LITE_RUNTIME;`)
- recompile: `protoc onnx.proto --cpp_out=src`


Usage
-----

The build creates `onnx2c` binary. 
Run 

	`./onnx2c [your ONNX model file] > model.c`

At the end of the `model.c` there is a function called 'void entry(...)'.
Call that from your main model. Function parameters are named as in your ONNX model.


Performance
-----------

or, how to extrapolate from incomplete data.

At the time of writing this, a single ONNX neural net has been benchmarked with
onnx2c - the ["Hello World"-sine generating example from TensorFlow Lite micro](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb) and compiled to ONNX with keras2onnx.

The same ONNX file was compiled with STM32CubeAI and onnx2c to a STM32F411
running STM32Cube HAL with a clock speed of 96MHz. With same project and optimization settings 
(gcc -O4), measuring inference time by toggling GPIO pins, the STMCubeAI-generated version
ran at 3ms, while the onnx2c one took 0.119ms.

The same NN model was measured [on a youtube video by Shawn Hymel](https://www.youtube.com/watch?v=crJcDqIUbP4),
run by TFL and STM32CubeAI (but in contrast, compiled from TFL, not ONNX as above - not sure if this matters).
The device used was a STM32L4 at 80MHz.
There the TFL version took 104us, while the STM32CubeAI one took 74us. These measurements were done by timer counting.

Finally, running the onnx2c created version on the same STM32F411, but with libopencm3 and at 84MHz, the execution time
dropped to 25us.

Memory consumption was about similar:
| platform        |text      |  data  |  bss | runtime |
|:----------------|---------:|-------:|-----:|--------:|
|STM HAL + onnx2c |     8276 |   1300 |  3060| 119us   |
|STM HAL + CubeAI |     14372|   1696 |  2808| 3000us  |
|OpenCM3 + onnx2c |     8236 |   1296 |   388| 25us    |

Here I guess CubeAI has full set of ONNX operators linked in, which explains
the bigger text size. I have no explanation why the same onnx2c output code,
compiled with same gcc (gcc-arm-none-eabi-8-2019-q3-update, from ARM) and same
target board shows so distinct differnces in run time.

More datapoints are definitely needed...



