set(BENCHMARK_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR})


# Run the benchmark tests separately to check they calculate correct values.
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

# Link all benchmark tests together
add_library(all_onnx2c_benchmarks
	conv_yolov6n_inputlayer.c
	conv_yolov6n_biggestconv.c
	conv_yolov6n_lastconv.c
)

# Create the benchmark binary
add_executable(onnx2c_benchmark benchmark_main.cc)
target_link_libraries(onnx2c_benchmark benchmark all_onnx2c_benchmarks)
target_compile_options(onnx2c_benchmark
	PUBLIC
		-I${CMAKE_CURRENT_BINARY_DIR}/
	)

# The target to run the benchmark suite. See benchmark_helper.sh for more documentation.
add_custom_target(run_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_helper.sh
	DEPENDS onnx2c_benchmark)

