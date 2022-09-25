# Create the benchmark binary
add_executable(onnx2c_benchmark benchmark_main.cc)
target_link_libraries(onnx2c_benchmark benchmark)
add_dependencies(onnx2c_benchmark dummy) # depend on the sources being created, but don't link - main.cc #includes the sources
target_compile_options(onnx2c_benchmark
	PUBLIC
		-I${CMAKE_CURRENT_BINARY_DIR}/..
	)

# The target to run the benchmark suite. See benchmark_helper.sh for more documentation.
add_custom_target(run_benchmark
	COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_helper.sh
	DEPENDS onnx2c_benchmark)

