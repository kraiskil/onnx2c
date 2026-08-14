set(BENCHMARK_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR})


# Run the benchmark tests separately to check they calculate correct values.
# These get run as part of 'make test' (i.e. running CTest)
# As a side effect, the benchmark models get complied to C sources.
function( onnx2c_benchmark node_name)
	compile_onnx( ${BENCHMARK_TEST_DATA_DIR}/benchmark_${node_name}/model.onnx ${node_name}_none.c -p none)
	compile_onnx( ${BENCHMARK_TEST_DATA_DIR}/benchmark_${node_name}/model.onnx ${node_name}_all.c -p im2col_all)
	compile_onnx( ${BENCHMARK_TEST_DATA_DIR}/benchmark_${node_name}/model.onnx ${node_name}_heuristic.c -p im2col)
	ONNX_type_test(
			${node_name}
			${BENCHMARK_TEST_DATA_DIR}/benchmark_${node_name}
			benchmark_${node_name}
			0.0002
			0
	)
endfunction()
onnx2c_benchmark(conv_yolov6n_inputlayer)
onnx2c_benchmark(conv_yolov6n_biggestconv)
onnx2c_benchmark(conv_yolov6n_lastconv)
onnx2c_benchmark(conv_fits_128k)
onnx2c_benchmark(conv_resnet_stem_7x7)
onnx2c_benchmark(conv_resnet_3x3)
onnx2c_benchmark(conv_resnet_bottleneck_1x1)
onnx2c_benchmark(conv_mobilenet_depthwise_3x3)
onnx2c_benchmark(conv_mobilenet_pointwise_1x1)

# add a dummy target to which the onnx2c generated files (1st line in onnx2c_benchmark())
# get linked into. This library is not used - it only serves as a target to force
# the generation of the benchmark C versions of the benchmark tests.
add_library(dummy
	conv_yolov6n_inputlayer_none.c
	conv_yolov6n_inputlayer_all.c
	conv_yolov6n_inputlayer_heuristic.c
	conv_yolov6n_biggestconv_none.c
	conv_yolov6n_biggestconv_all.c
	conv_yolov6n_biggestconv_heuristic.c
	conv_yolov6n_lastconv_none.c
	conv_yolov6n_lastconv_all.c
	conv_yolov6n_lastconv_heuristic.c
	conv_fits_128k_none.c
	conv_fits_128k_all.c
	conv_fits_128k_heuristic.c
	conv_resnet_stem_7x7_none.c
	conv_resnet_stem_7x7_all.c
	conv_resnet_stem_7x7_heuristic.c
	conv_resnet_3x3_none.c
	conv_resnet_3x3_all.c
	conv_resnet_3x3_heuristic.c
	conv_resnet_bottleneck_1x1_none.c
	conv_resnet_bottleneck_1x1_all.c
	conv_resnet_bottleneck_1x1_heuristic.c
	conv_mobilenet_depthwise_3x3_none.c
	conv_mobilenet_depthwise_3x3_all.c
	conv_mobilenet_depthwise_3x3_heuristic.c
	conv_mobilenet_pointwise_1x1_none.c
	conv_mobilenet_pointwise_1x1_all.c
	conv_mobilenet_pointwise_1x1_heuristic.c
)

# Run the on-host benchmarking.
# See also the target benchmarking folder for MCU tests.
add_subdirectory(host)

add_custom_target(
	run_target_stm32f411_benchmarks
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/target_stm32f411_nucleo/run_benchmarks.sh
	${CMAKE_SOURCE_DIR}
	)
