
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNX_BACKEND_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/onnx/backend/test/data/)
set(ONNX_NODE_TEST_DATA_DIR ${ONNX_BACKEND_TEST_DATA_DIR}/node/)

add_executable( testrun
	onnx_backend_tests_runner.cc)
target_link_libraries(testrun onnx2c_lib protobuf)
target_compile_options(testrun
	PRIVATE
		-I${CMAKE_CURRENT_SOURCE_DIR}/../aixlog/include
	)

function( compile_onnx onnx_file c_file )
	add_custom_command(
		OUTPUT
			${c_file}
		COMMAND
			onnx2c ${onnx_file} > ${c_file}
		DEPENDS 
			${onnx_file}
			onnx2c
	) 
endfunction()


function( ONNX_type_test node_name data_dir test_ctest_name accuracy test_data_set)

	add_custom_command(
		OUTPUT
		${node_name}_${test_data_set}_generated.c
		COMMAND
		testrun ${data_dir} ${accuracy} ${test_data_set} > ${node_name}_${test_data_set}_generated.c
		DEPENDS
		#TODO also depends on test data -> don't depend, always run
		testrun
		)

	add_executable( ${node_name}_${test_data_set}_test
		${node_name}_${test_data_set}_generated.c
		)
	target_compile_options( ${node_name}_${test_data_set}_test
		PRIVATE
			-Wall -Werror
			#TODO: space for output tensor is generated, but not used.
			-Wno-unused-variable
		)
	target_link_libraries( ${node_name}_${test_data_set}_test m )

	add_test( ${test_ctest_name}
		${node_name}_${test_data_set}_test
		)
endfunction()

function( ONNX_backend_node_test node_name)
	ONNX_type_test(
		${node_name}
		${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
		ONNX_backend_${node_name}
		0.00002
		0
	)
endfunction()

ONNX_backend_node_test(add)
ONNX_backend_node_test(add_bcast)

ONNX_backend_node_test(batchnorm_epsilon)
ONNX_backend_node_test(batchnorm_example)

ONNX_backend_node_test(conv_with_strides_no_padding)
ONNX_backend_node_test(conv_with_strides_padding)
ONNX_backend_node_test(conv_with_strides_and_asymmetric_padding)
ONNX_type_test(operator_conv ${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-operator/test_operator_conv ONNX_backend_pytorch_conv 0.00001 0)

ONNX_backend_node_test(flatten_axis0)
ONNX_backend_node_test(flatten_axis1)
ONNX_backend_node_test(flatten_axis2)
ONNX_backend_node_test(flatten_axis3)
ONNX_backend_node_test(flatten_default_axis)
ONNX_backend_node_test(flatten_negative_axis1)
ONNX_backend_node_test(flatten_negative_axis2)
ONNX_backend_node_test(flatten_negative_axis3)
ONNX_backend_node_test(flatten_negative_axis4)

ONNX_backend_node_test(lstm_defaults)
ONNX_backend_node_test(lstm_with_initial_bias)
ONNX_backend_node_test(lstm_with_peepholes)

ONNX_backend_node_test(matmul_2d)
# only 2d matmul implemented. _integer has multiple outputs
#ONNX_backend_node_test(matmul_3d)
#ONNX_backend_node_test(matmul_4d)


ONNX_backend_node_test(maxpool_1d_default)
ONNX_backend_node_test(maxpool_2d_precomputed_pads)
ONNX_backend_node_test(maxpool_2d_strides)
# This uses implicit padding on the axis ends which
# is agianst specifications - might be fragile
# see https://github.com/onnx/onnx/issues/2971
# and https://github.com/onnx/onnx/issues/2927
ONNX_backend_node_test(maxpool_2d_ceil)
ONNX_backend_node_test(maxpool_2d_precomputed_same_upper)
ONNX_backend_node_test(maxpool_2d_uint8)
ONNX_backend_node_test(maxpool_2d_default)
ONNX_backend_node_test(maxpool_2d_precomputed_strides)
ONNX_backend_node_test(maxpool_3d_default)
ONNX_backend_node_test(maxpool_2d_dilations)
ONNX_backend_node_test(maxpool_2d_same_lower)
ONNX_backend_node_test(maxpool_2d_pads)
ONNX_backend_node_test(maxpool_2d_same_upper)
ONNX_backend_node_test(maxpool_with_argmax_2d_precomputed_pads)
#Has column-major order
#ONNX_backend_node_test(maxpool_with_argmax_2d_precomputed_strides)


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

ONNX_backend_node_test(sigmoid)
ONNX_backend_node_test(sigmoid_example)

ONNX_backend_node_test(squeeze)
ONNX_backend_node_test(squeeze_negative_axes)

ONNX_backend_node_test(softmax_axis_0)
ONNX_backend_node_test(softmax_axis_1)
ONNX_backend_node_test(softmax_axis_2)
ONNX_backend_node_test(softmax_default_axis)
ONNX_backend_node_test(softmax_example)
ONNX_backend_node_test(softmax_large_number)
ONNX_backend_node_test(softmax_negative_axis)

ONNX_backend_node_test(transpose_default)
ONNX_backend_node_test(transpose_all_permutations_0)
ONNX_backend_node_test(transpose_all_permutations_1)
ONNX_backend_node_test(transpose_all_permutations_2)
ONNX_backend_node_test(transpose_all_permutations_3)
ONNX_backend_node_test(transpose_all_permutations_4)
ONNX_backend_node_test(transpose_all_permutations_5)

# More end-to-end kind of tests here
add_subdirectory(tfl_helloworld)
add_subdirectory(mnist)
add_subdirectory(velardo)
add_subdirectory(simple_networks)
