# CMake generated Testfile for 
# Source directory: /Users/yjack/GitHub/andes/onnx2c/test/benchmarks
# Build directory: /Users/yjack/GitHub/andes/onnx2c/build/test/benchmarks
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(benchmark_conv_yolov6n_inputlayer "conv_yolov6n_inputlayer_0_test")
set_tests_properties(benchmark_conv_yolov6n_inputlayer PROPERTIES  _BACKTRACE_TRIPLES "/Users/yjack/GitHub/andes/onnx2c/test/CMakeLists.txt;58;add_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;9;ONNX_type_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;17;onnx2c_benchmark;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;0;")
add_test(benchmark_conv_yolov6n_biggestconv "conv_yolov6n_biggestconv_0_test")
set_tests_properties(benchmark_conv_yolov6n_biggestconv PROPERTIES  _BACKTRACE_TRIPLES "/Users/yjack/GitHub/andes/onnx2c/test/CMakeLists.txt;58;add_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;9;ONNX_type_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;18;onnx2c_benchmark;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;0;")
add_test(benchmark_conv_yolov6n_lastconv "conv_yolov6n_lastconv_0_test")
set_tests_properties(benchmark_conv_yolov6n_lastconv PROPERTIES  _BACKTRACE_TRIPLES "/Users/yjack/GitHub/andes/onnx2c/test/CMakeLists.txt;58;add_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;9;ONNX_type_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;19;onnx2c_benchmark;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;0;")
add_test(benchmark_conv_fits_128k "conv_fits_128k_0_test")
set_tests_properties(benchmark_conv_fits_128k PROPERTIES  _BACKTRACE_TRIPLES "/Users/yjack/GitHub/andes/onnx2c/test/CMakeLists.txt;58;add_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;9;ONNX_type_test;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;20;onnx2c_benchmark;/Users/yjack/GitHub/andes/onnx2c/test/benchmarks/CMakeLists.txt;0;")
subdirs("host")
