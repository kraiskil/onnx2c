Developping onnx2c
==================

Adding a new node/op implementation
-----------------------------------

If you are adding the implementation for a new ONNX operator:

- create the implementation file for it in `src/nodes/` (check out
  `src/nodes/TEMPLATE` for a starting point).
  (At the begining of the project implementations were all put into
   a .h file, but with the multitude of nodes, the compilation
   times slow down, so prefer using both .h and .cc)
- add a case to the switch in `Graph::findNode()` in `src/graph.cc`
  and an `#include` in the lines above that function
- add the new file to CMakeLists.txt
- add onnx backed tests for the new node in `test/CMakeLists.txt`.
  Search for them in `onnx/onnx/backend/test/data/node`.


Testing
-------

The main onnx2c testing is built ontop of the ONNX backend node tests.

On top of these there are two classes of benchmarking tests in the test suite

 * Google Benchmark based. This is the new version.
 * ONNX model zoo based. Old, and less useful.

And there is a limited support to test performance on top of embedded targets.

### Unit tests

The onnx2c build runs onnx backend tests as unit/acceptance tests.


To run these, continue the build steps with:
```
make
make test
```

All test should and must pass.

### Google Benchmark based tests

The benchmark binary is built in `test/benchmarks` as a part of the unit test framework.

Run it by executing the fake custom target `run_benchmarks` (e.g. `make run_benchmarks`).

The `run_benchmarks` is intended as a development tool. It is useful only when chaninging the
generated code from those operators/nodes that are included in the benchmark suite.

Note, `run_benchmarks` is host computer specific, and must be first run with a clean master build
to get a reference baseline. See the comments in `test/benchmarks/host/benchmark_helper.sh` for more info.

### ONNX model zoo based tests

These are mostly deprecated, but the infrastructure is still left in place.

Run:
```
cd tests/onnx_model_zoo
./donwload.sh
```
and continue with a fresh build (i.e. re-run `cmake`).

Included are implementations of e.g. Squeezenet and Alexnet. Some of these take minutes to compile, so
they are mostly interesting for onnx2c development.


### On target testing

There is a build system in the `scripts` folder to complile and run any `.onnx` file on an embedded
development board.
It assumes [LibOpenCM3](https://http://libopencm3.org/), a `arm-none-eabi-gcc` compiler chain, and
[OpenOCD](https://openocd.org/) are available in `PATH` and the `OPENCM3_DIR` environment variable.


For now, the script hard-codes STM32F411 NUCLEO as the target (pull requests to remedy this welcome!).

To test if your own network file will fit, run `scripts/measure_stm32f411_nucleo.sh [file.onnx]`.
This will compile, flash and report memory usage and runtime for the `.onnx` file. No checks of correctnes
are made.

A collection of benchmarking tests can be run with `make run_target_stm32f411_benchmarks`.

Coding style
------------

 - indent with tab, align with space
 - run `./scripts/code_layout_diff.sh` before pushing.

CI doe not enforce a coding style yet. 
