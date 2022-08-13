onnx2c
======

Onnx2c is a [ONNX](https://onnx.ai) to C compiler. It will read an ONNX file,
and generate C code to be included in your project.

Onnx2c's target is "Tiny ML", meaning running the inference on microcontrollers. To make this
easier, the generated code:

- Does not `#include <stdio.h>` (i.e. no `printf()`s)
- Compile-time allocates buffers. Does not use dynamic memory allocation or (much) stack memory
- Has no library requirements except standard C maths library. (Floating point hardware recommended!)
- Should be compiler-friendly allowing the C compiler optimize the output as well as it can
- Is contained in one single C file for easier project management

The idea behind onnx2c is to be an easy-to-use tool with no learning curve. If you can export your trained
neural network to an ONNX file (e.g. PyTorch and Tensorflow both can) and you have a working microcontroller
project, then joining the two with onnx2c should be easy.

To make all of the above easier to achieve, there are some non-goals for onnx2c:

 - ONNX specification coverage. (For now, 91 out of 166 ONNX Operands are at least partially implemented).
 - accelerators
 - backpropagation (i.e. training)


Building
--------

Make sure you have ProtocolBuffers libraries installed, e.g.: 

 - Ubuntu: `apt install libprotobuf-dev protobuf-compiler`
 - MacOS: `brew install protobuf`

Get the sources:

```
git clone https://github.com/kraiskil/onnx2c.git
cd onnx2c
git submodule update --init
```

then run a standard CMake build

```
mkdir build
cd build
cmake ..
make onnx2c
```


### Getting `error: ‘class onnx::ModelProto’ has no member named ‘ParseFromIstream’;` ?

If you have ProtoBuf 3.6 or earlier, you need the following modification to `onnx/onnx/onnx.proto`

- remove the last lines (i.e. option `optimize_for = LITE_RUNTIME;`)

With ProtoBuf 3.12 (e.g. Ubuntu 20.10 onwards) this modification is not needed.

Versions between 3.6 and 3.12 are uninvestigated.


### Unit tests

The onnx2c build runs onnx backend tests as unit/acceptance tests.


To run these, continue the build steps with:
```
make
make test
```

On top of these there are two classes of benchmarking tests in the test suite

 * Google Benchmark based. This is the new version.
 * ONNX model zoo based. Old, and less useful.

#### Google Benchmark based tests

The benchmark binary is built in `test/benchmarks` as a part of the unit test framework.

Run it by executing the fake custom target `run_benchmarks` (e.g. `make run_benchmarks`).

The `run_benchmarks` is intended as a development tool. It is useful only when chaninging the
generated code from those operators/nodes that are included in the benchmark suite.

Note, `run_benchmarks` is host computer specific, and must be first run with a clean master build
to get a reference baseline. See the comments in `test/benchmarks/benchmark_helper.sh` for more info.

#### ONNX model zoo based tests

These are mostly deprecated, but the infrastructure is still left in place.

Run:
```
cd tests/onnx_model_zoo
./donwload.sh
```
and continue with a fresh build (i.e. re-run `cmake`).

Included are implementations of e.g. Squeezenet and Alexnet. Some of these take minutes to compile, so
they are mostly interesting for onnx2c development.


Usage
-----

The build creates `onnx2c` binary. 
Run 

`./onnx2c [your ONNX model file] > model.c`

At the end of the `model.c` there is a function called 'void entry(...)'.
Call that from your main program to run inference. Function parameters are named as in your ONNX model.

Using the compiler `-ffast-math` (or equivalent) when compiling onnx2c-generated code increases computation speed.
See the [GCC wiki on floating point maths](https://gcc.gnu.org/wiki/FloatingPointMath) for details.

Onnx2c has an [experimental quantization option](quantization.md) to convert floating point calculation to integers.

`./onnx2c -h` prints out all available command line options.


On-target performance
---------------------

or, how to extrapolate from incomplete data.

At the time of writing this, a single ONNX neural net has been benchmarked with
onnx2c - the ["Hello World"-sine generating example from TensorFlow Lite micro](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb) and compiled to ONNX with keras2onnx.

That ONNX file was compiled with  STM32CubeAI and onnx2c to a STM32F411
running STM32Cube HAL with a clock speed of 84 or 96MHz. With same project and
optimization settings (gcc -O4), measuring inference time by toggling GPIO pins,
the STMCubeAI-generated version ran at 490us, while the onnx2c one took 20us.

See Notes below for a description of the RAM optmimized version.

Memory consumption was about similar:
| platform               |text      |  data  |  bss | runtime |
|:-----------------------|---------:|-------:|-----:|--------:|
|STM HAL + onnx2c @96MHz |     8276 |   1300 |  3060| 20us    |
|STM HAL + CubeAI @96MHz |     14372|   1696 |  2808| 490us   |
|OpenCM3 + onnx2c @84MHz |     8236 |   1296 |   388| 25us    |
|--"-- (onnx2c RAM opt)  |     8236 |     12 |   388| 29us    |


### Comparison 

The same NN model was measured
[on a youtube video by Shawn Hymel](https://www.youtube.com/watch?v=crJcDqIUbP4),
run both via TFL and STM32CubeAI. The device used was a STM32L4 at 80MHz.
There the TFL version took 104us, while the STM32CubeAI one took 74us.

The STM32L4 used by Hymel is a low-power version of the STM32F4, so the L4 
certainly should not be faster than the F4. Same versions of CubeAI were used.
The only difference was that Hymel fed the TFL model to CubeAI, not the ONNX model
as in the above measurement. I am not sure if this is relevant, but so far
it is the only think I can think of that could explain the difference.
Also the measured ONNX model was not converted from the TFL model that Hymel used,
but re-trained using the tutorial. But this most likely is not the cause for the
execution speed difference.

More datapoints are definitely needed...

### Notes

The above values are made with an older version of onnx2c. Later versions
have added a "mark constant tensors as 'const'" optimisation, that significantly
reduces RAM usage, but has a small performance penalty (4us in the above case).

This is because when marked const, GCC generates code that reads the 'const' vectors
from flash (as opposed to copying them to RAM). Reading flash is, of course,
slower than RAM.

Disabling of this optimisation should be added as a command-line option to onnx2c.

