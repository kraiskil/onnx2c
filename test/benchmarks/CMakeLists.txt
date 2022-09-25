set(BENCHMARK_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR})


# Run the benchmark tests separately to check they calculate correct values.
# These get run as part of 'make test' (i.e. running CTest)
# As a side effect, the benchmark models get complied to C sources.
function( onnx2c_benchmark node_name)
	compile_onnx( ${BENCHMARK_TEST_DATA_DIR}/benchmark_${node_name}/model.onnx ${node_name}.c)
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

# add a dummy target to which the onnx2c generated files (1st line in onnx2c_benchmark())
# get linked into. This library is not used - it only serves as a target to force
# the generation of the benchmark C versions of the benchmark tests.
add_library(dummy
	conv_yolov6n_inputlayer.c
	conv_yolov6n_biggestconv.c
	conv_yolov6n_lastconv.c
	conv_fits_128k.c
)

# Run the on-host benchmarking.
# See also the target benchmarking folder for MCU tests.
add_subdirectory(host)

add_custom_target(
	run_target_stm32f411_benchmarks
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/target_stm32f411_nucleo/run_benchmarks.sh
	${CMAKE_SOURCE_DIR}
	)
