Developping onnx2c
==================

Adding a new node/op implementation
-----------------------------------

If you are adding the implementation for a ONNX operator:

- create the implementation file for it in `src/nodes/`
- add a case to the switch in `Graph::findNode()` in `src/graph.cc`
- add onnx backed tests for the new node in `test/CMakeLists.txt`


Protobuffer lite
----------------

Some versions (on Ubuntu 16.04 at least), the generated protobuffer
files would not compile without the following modification to the file
`onnx/onnx/onnx.proto`

- remove the last lines (i.e. option `optimize_for = LITE_RUNTIME;`)

On newer vesions (Ubuntu 20.10) this is not needed.


Testing
-------

Onnx2c has functional correctness testing built with CTest. Run `make test`
to execute this test suite. All test should and must pass.

Besides this, there is are a few crude performance tests that can be run with
`make perftest_run`. These tests measure the execution time of compiled onnx models
using unix `time` utility. This means the result will be different on different computers,
and you should only pay attention to the difference in execution time between git
revisions. Also check that the system load is low, or the execution times will be useless.

Better performance test frameworks are sorely needed...


Coding style
------------

 - indent with tab, align with space

