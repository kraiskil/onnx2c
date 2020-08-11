add_subdirectory(tfl_helloworld)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNX_NODE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/onnx/backend/test/data/node/)

add_executable( testrun
	onnx_backend_tests_runner.cc)
target_link_libraries(testrun onnx2c_lib protobuf)


function( ONNX_backend_node_test node_name)

	add_custom_command(
		OUTPUT
		${node_name}_generated.c
		COMMAND
		testrun ${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}/ > ${node_name}_generated.c
		DEPENDS
		#TODO also depends on test data -> don't depend, always run
		testrun
		)

	add_executable( ${node_name}_test
		${node_name}_generated.c
		)

	add_test( ONNX_backend_${node_name}
		${node_name}_test
		)
endfunction()

ONNX_backend_node_test(add)
ONNX_backend_node_test(add_bcast)
ONNX_backend_node_test(matmul_2d)
ONNX_backend_node_test(relu)
ONNX_backend_node_test(reshape_extended_dims)
#ONNX_backend_node_test(reshape_negative_extended_dims)  # zero and negative dimensions TBD
ONNX_backend_node_test(reshape_reduced_dims)
ONNX_backend_node_test(reshape_reordered_last_dims)
#ONNX_backend_node_test(reshape_zero_dim)
#ONNX_backend_node_test(reshape_negative_dim)
ONNX_backend_node_test(reshape_one_dim)
ONNX_backend_node_test(reshape_reordered_all_dims)
#ONNX_backend_node_test(reshape_zero_and_negative_dim)

