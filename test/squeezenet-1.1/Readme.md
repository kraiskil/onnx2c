This is SqueezeNet 1.1 implmenetation from the [ONNX model zoo](https://github.com/onnx/models).
License: MIT

SqueezeNet is rather heavy to compile to C and to binary, so it is not part of the default test suite.
It is used as an execution performance benchmark instead.

The [Squeezenet Readme](https://github.com/onnx/models/tree/master/vision/classification/squeezenet)
says:
 
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0

This is not visible when running the onnx2c-generated SqueezeNet binaries. Might be most of the
measured execution time is taken up by overheads, and the time to run inference is not noticable.
What exactly is happening here should be investigated.
