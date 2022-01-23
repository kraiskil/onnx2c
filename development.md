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

Onnx2c has functional correctness testing built with CTest. Run `make test`
to execute this test suite. All test should and must pass.

There are a few bigger networks taken from the ONNX model zoo that can be used
as timing performance checks. Before running CMake, cd into `test/onnx_model_zoo/`
and execute `download.sh`. CMake then picks up those tests that have their `.tar.gz`
extracted.

Beware, some of these tests compile for minutes!

Better performance test frameworks are sorely needed...


Coding style
------------

 - indent with tab, align with space

