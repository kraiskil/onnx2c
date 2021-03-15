Quantization
============

Onnx2c has an option to quantize the generated code.
Quantization in this context means transforming computation
from floating point to integer arithmentic. And in the context
of onnx2c in particular integer arithmetic means 8-bit integers.


Usage:
------
add the `-quantize` comand line option

    onnx2c -quantize <inputmodel.onnx>


Limitations:
------------
Onnx2c quantization is very much in "alpha" stage.
Only a select few Operators allow for quantization,
and even with those the quantization is a direct linear
scaling of the tensors to the integer range.

More thorough implementation would run something like the
BatchNormalization after a quantized node to get an optimal
"dynamic range" for the tensors.


Related work:
-------------
At least pytorch and onnxruntime have quantization tools available.
These work "in the frontend" of the building of a quantized network:
either during training or on the onnx intermediate file.

In contrast, onnx2c quantization works with "traditional" floating point
onnx files, and does the quantization "in the backend".

Such front-end quantized networks should of course work with onnx2c too.
No `-quantize` option should be given when compiling such quantized networks.

TODO: add links to the onnxruntime and pytorch quantization tools.
