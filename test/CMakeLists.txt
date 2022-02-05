
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)
set(ONNX_BACKEND_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../onnx/onnx/backend/test/data/)
set(ONNX_NODE_TEST_DATA_DIR ${ONNX_BACKEND_TEST_DATA_DIR}/node/)

# testgen utility that reads the input from a
# onnx "standard" formatted test (see directory $ONNX_BACKEND_TEST_DATA_DIR)
# and generates input, the network-under-test, expected output and a main()
# to check output matches expected output.
add_executable( testgen
	onnx_backend_tests_generator.cc)
target_link_libraries(testgen onnx2c_lib ${Protobuf_LIBRARIES})
target_compile_options(testgen
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


function( ONNX_type_test_build node_name data_dir accuracy test_data_set)

	add_custom_command(
		OUTPUT
		${node_name}_${test_data_set}_generated.c
		COMMAND
		testgen ${data_dir} ${accuracy} ${test_data_set} > ${node_name}_${test_data_set}_generated.c
		DEPENDS
		#TODO also depends on test data -> don't depend, always run
		testgen
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

endfunction()

function( ONNX_type_test node_name data_dir test_ctest_name accuracy test_data_set)
	ONNX_type_test_build(${node_name} ${data_dir} ${accuracy} ${test_data_set})
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
function( ONNX_backend_pytorch_converted_test node_name)
	ONNX_type_test(
		${node_name}
		${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-converted/test_${node_name}
		ONNX_backend_${node_name}
		0.00002
		0
	)
endfunction()


function( ONNX_backend_node_test_with_accuracy node_name accuracy)
	ONNX_type_test(
		${node_name}
		${ONNX_NODE_TEST_DATA_DIR}/test_${node_name}
		ONNX_backend_${node_name}
		${accuracy}
		0
	)
endfunction()


set(ONNX_LOCAL_NODE_TEST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/local_ops/)
function( local_node_test node_name)
	ONNX_type_test(
			${node_name}
			${ONNX_LOCAL_NODE_TEST_DATA_DIR}/test_${node_name}
			local_node_${node_name}
			0.00002
			0
	)
endfunction()



ONNX_backend_node_test(abs)

ONNX_backend_node_test(acos)
ONNX_backend_node_test(acos_example)

ONNX_backend_node_test(acosh)
ONNX_backend_node_test(acosh_example)

ONNX_backend_node_test(add)
ONNX_backend_node_test(add_bcast)

ONNX_backend_node_test(and2d)
ONNX_backend_node_test(and4d)
ONNX_backend_node_test(and_bcast3v2d)
ONNX_backend_node_test(and_bcast4v3d)
ONNX_backend_node_test(and3d)
ONNX_backend_node_test(and_bcast3v1d)
ONNX_backend_node_test(and_bcast4v2d)
ONNX_backend_node_test(and_bcast4v4d)

ONNX_backend_node_test(asin)
ONNX_backend_node_test(asin_example)

ONNX_backend_node_test(asinh)
ONNX_backend_node_test(asinh_example)

ONNX_backend_node_test(atan)
ONNX_backend_node_test(atan_example)

ONNX_backend_node_test(atanh)
ONNX_backend_node_test(atanh_example)

ONNX_backend_node_test(averagepool_1d_default)
ONNX_backend_node_test(averagepool_2d_precomputed_pads)
ONNX_backend_node_test(averagepool_2d_same_upper)
ONNX_backend_node_test(averagepool_2d_ceil)
ONNX_backend_node_test(averagepool_2d_precomputed_pads_count_include_pad)
ONNX_backend_node_test(averagepool_2d_strides)
ONNX_backend_node_test(averagepool_2d_default)
ONNX_backend_node_test(averagepool_2d_precomputed_same_upper)
ONNX_backend_node_test(averagepool_3d_default)
ONNX_backend_node_test(averagepool_2d_pads)
ONNX_backend_node_test(averagepool_2d_precomputed_strides)
ONNX_backend_node_test(averagepool_2d_pads_count_include_pad)
ONNX_backend_node_test(averagepool_2d_same_lower)

ONNX_backend_node_test(batchnorm_epsilon)
ONNX_backend_node_test(batchnorm_example)

ONNX_backend_node_test(bitshift_left_uint16)
ONNX_backend_node_test(bitshift_left_uint64)
ONNX_backend_node_test(bitshift_right_uint16)
ONNX_backend_node_test(bitshift_right_uint64)
ONNX_backend_node_test(bitshift_left_uint32)
ONNX_backend_node_test(bitshift_left_uint8)
ONNX_backend_node_test(bitshift_right_uint32)
ONNX_backend_node_test(bitshift_right_uint8)

#ONNX_backend_node_test(cast_BFLOAT16_to_FLOAT)
#ONNX_backend_node_test(cast_FLOAT16_to_DOUBLE)
ONNX_backend_node_test(cast_FLOAT_to_DOUBLE)
#ONNX_backend_node_test(cast_STRING_to_FLOAT)
ONNX_backend_node_test(cast_DOUBLE_to_FLOAT)
#ONNX_backend_node_test(cast_FLOAT16_to_FLOAT)
#ONNX_backend_node_test(cast_FLOAT_to_FLOAT16)
#ONNX_backend_node_test(cast_DOUBLE_to_FLOAT16)
#ONNX_backend_node_test(cast_FLOAT_to_BFLOAT16)
#ONNX_backend_node_test(cast_FLOAT_to_STRING)

ONNX_backend_node_test(ceil)
ONNX_backend_node_test(ceil_example)

ONNX_backend_node_test(celu)
# This does not test Celu
ONNX_backend_node_test(celu_expanded)

ONNX_backend_node_test(clip)
ONNX_backend_node_test(clip_default_int8_max)
ONNX_backend_node_test(clip_default_min)
ONNX_backend_node_test(clip_outbounds)
ONNX_backend_node_test(clip_default_inbounds)
ONNX_backend_node_test(clip_default_int8_min)
ONNX_backend_node_test(clip_example)
ONNX_backend_node_test(clip_splitbounds)
ONNX_backend_node_test(clip_default_int8_inbounds)
ONNX_backend_node_test(clip_default_max)
ONNX_backend_node_test(clip_inbounds)

ONNX_backend_node_test(concat_1d_axis_0)
ONNX_backend_node_test(concat_1d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_0)
ONNX_backend_node_test(concat_2d_axis_1)
ONNX_backend_node_test(concat_2d_axis_negative_1)
ONNX_backend_node_test(concat_2d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_0)
ONNX_backend_node_test(concat_3d_axis_1)
ONNX_backend_node_test(concat_3d_axis_2)
ONNX_backend_node_test(concat_3d_axis_negative_1)
ONNX_backend_node_test(concat_3d_axis_negative_2)
ONNX_backend_node_test(concat_3d_axis_negative_3)
ONNX_type_test(operator_concat2 ${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-operator/test_operator_concat2 ONNX_backend_pytorch_conv 0.00001 0)

ONNX_backend_node_test(constant)

ONNX_backend_node_test(cos)
ONNX_backend_node_test(cos_example)

ONNX_backend_node_test(cosh)
ONNX_backend_node_test(cosh_example)

ONNX_backend_node_test(conv_with_strides_no_padding)
ONNX_backend_node_test(conv_with_strides_padding)
ONNX_backend_node_test(conv_with_strides_and_asymmetric_padding)
ONNX_type_test(operator_conv ${ONNX_BACKEND_TEST_DATA_DIR}/pytorch-operator/test_operator_conv ONNX_backend_pytorch_conv 0.00001 0)

ONNX_backend_pytorch_converted_test(Conv1d)
ONNX_backend_pytorch_converted_test(Conv1d_groups)
ONNX_backend_pytorch_converted_test(Conv1d_pad1size1)
ONNX_backend_pytorch_converted_test(Conv1d_pad2size1)
ONNX_backend_pytorch_converted_test(Conv1d_dilated)
ONNX_backend_pytorch_converted_test(Conv1d_pad1)
ONNX_backend_pytorch_converted_test(Conv1d_pad2)
ONNX_backend_pytorch_converted_test(Conv1d_stride)

ONNX_backend_pytorch_converted_test(Conv2d)
ONNX_backend_pytorch_converted_test(Conv2d_depthwise)
ONNX_backend_pytorch_converted_test(Conv2d_depthwise_padded)
ONNX_backend_pytorch_converted_test(Conv2d_depthwise_strided)
ONNX_backend_pytorch_converted_test(Conv2d_depthwise_with_multiplier)
ONNX_backend_pytorch_converted_test(Conv2d_dilated)
ONNX_backend_pytorch_converted_test(Conv2d_groups)
ONNX_backend_pytorch_converted_test(Conv2d_groups_thnn)
ONNX_backend_pytorch_converted_test(Conv2d_no_bias)
ONNX_backend_pytorch_converted_test(Conv2d_padding)
ONNX_backend_pytorch_converted_test(Conv2d_strided)

ONNX_backend_pytorch_converted_test(Conv3d)
ONNX_backend_pytorch_converted_test(Conv3d_dilated_strided)
ONNX_backend_pytorch_converted_test(Conv3d_no_bias)
ONNX_backend_pytorch_converted_test(Conv3d_stride_padding)
ONNX_backend_pytorch_converted_test(Conv3d_dilated)
ONNX_backend_pytorch_converted_test(Conv3d_groups)
ONNX_backend_pytorch_converted_test(Conv3d_stride)

ONNX_backend_node_test(convinteger_with_padding)

ONNX_backend_node_test(div)
ONNX_backend_node_test(div_bcast)
ONNX_backend_node_test(div_example)

ONNX_backend_node_test(dropout_default)
ONNX_backend_node_test(dropout_default_mask_ratio)
ONNX_backend_node_test(dropout_default_ratio)
ONNX_backend_node_test(dropout_default_mask)
ONNX_backend_node_test(dropout_default_old)
ONNX_backend_node_test(dropout_random_old)

ONNX_backend_node_test(dynamicquantizelinear)
ONNX_backend_node_test(dynamicquantizelinear_max_adjusted)
ONNX_backend_node_test(dynamicquantizelinear_min_adjusted)
#These are not unit tests for dynamicquantizelinear, despite the name
#ONNX_backend_node_test(dynamicquantizelinear_max_adjusted_expanded)
#ONNX_backend_node_test(dynamicquantizelinear_min_adjusted_expanded)
#ONNX_backend_node_test(dynamicquantizelinear_expanded)

ONNX_backend_node_test(elu)
ONNX_backend_node_test(elu_default)
ONNX_backend_node_test(elu_example)

ONNX_backend_node_test(equal)
ONNX_backend_node_test(equal_bcast)

ONNX_backend_node_test(erf)

ONNX_backend_node_test(exp)
ONNX_backend_node_test(exp_example)

ONNX_backend_node_test(flatten_axis0)
ONNX_backend_node_test(flatten_axis1)
ONNX_backend_node_test(flatten_axis2)
ONNX_backend_node_test(flatten_axis3)
ONNX_backend_node_test(flatten_default_axis)
ONNX_backend_node_test(flatten_negative_axis1)
ONNX_backend_node_test(flatten_negative_axis2)
ONNX_backend_node_test(flatten_negative_axis3)
ONNX_backend_node_test(flatten_negative_axis4)

ONNX_backend_node_test(floor)
ONNX_backend_node_test(floor_example)

ONNX_backend_node_test(gather_0)
ONNX_backend_node_test(gather_1)
ONNX_backend_node_test(gather_negative_indices)
ONNX_backend_node_test(gather_2d_indices)

#ONNX_backend_node_test(gather_elements_1)
#ONNX_backend_node_test(gather_elements_0)
#ONNX_backend_node_test(gather_elements_negative_indices)

#ONNX_backend_node_test(gathernd_example_float32)
#ONNX_backend_node_test(gathernd_example_int32)
#ONNX_backend_node_test(gathernd_example_int32_batch_dim1)

ONNX_backend_node_test(gemm_all_attributes)
ONNX_backend_node_test(gemm_default_no_bias)
ONNX_backend_node_test(gemm_default_zero_bias)
ONNX_backend_node_test(gemm_alpha)
ONNX_backend_node_test(gemm_default_scalar_bias)
ONNX_backend_node_test(gemm_transposeA)
ONNX_backend_node_test(gemm_beta)
ONNX_backend_node_test(gemm_default_single_elem_vector_bias)
ONNX_backend_node_test(gemm_transposeB)
ONNX_backend_node_test(gemm_default_matrix_bias)
ONNX_backend_node_test(gemm_default_vector_bias)

local_node_test(gemm_CMx1)
local_node_test(gemm_CMxN)
local_node_test(gemm_C1xN)
local_node_test(gemm_C1x1)
local_node_test(gemm_C1)
local_node_test(gemm_CN)
local_node_test(gemm_CMx1_transA)
local_node_test(gemm_CMxN_transA)
local_node_test(gemm_C1xN_transA)
local_node_test(gemm_C1x1_transA)
local_node_test(gemm_C1_transA)
local_node_test(gemm_CN_transA)
local_node_test(gemm_C1_transB)
local_node_test(gemm_CN_transB)
local_node_test(gemm_CMxN_transB)
local_node_test(gemm_CMxN_transA_transB)
local_node_test(gemm_C1xN_transA_transB)
local_node_test(gemm_CMx1_transA_transB)
local_node_test(gemm_CN_transA_transB)

ONNX_backend_node_test(globalaveragepool)
ONNX_backend_node_test(globalaveragepool_precomputed)

ONNX_backend_node_test(greater)
ONNX_backend_node_test(greater_equal)
#ONNX_backend_node_test(greater_equal_bcast_expanded)
ONNX_backend_node_test(greater_bcast)
ONNX_backend_node_test(greater_equal_bcast)
#ONNX_backend_node_test(greater_equal_expanded)

ONNX_backend_node_test(hardsigmoid)
ONNX_backend_node_test(hardsigmoid_default)
ONNX_backend_node_test(hardsigmoid_example)

ONNX_backend_node_test(hardswish)
ONNX_backend_node_test(hardswish_expanded) #doesn't test hardswish

ONNX_backend_node_test(instancenorm_epsilon)
ONNX_backend_node_test(instancenorm_example)

ONNX_backend_node_test(leakyrelu)
ONNX_backend_node_test(leakyrelu_default)
ONNX_backend_node_test(leakyrelu_example)

ONNX_backend_node_test(less)
ONNX_backend_node_test(less_equal)
#ONNX_backend_node_test(less_equal_bcast_expanded)
ONNX_backend_node_test(less_bcast)
ONNX_backend_node_test(less_equal_bcast)
#ONNX_backend_node_test(less_equal_expanded)

ONNX_backend_node_test(log)
ONNX_backend_node_test(log_example)

ONNX_backend_node_test_with_accuracy(lrn 0.005)
ONNX_backend_node_test_with_accuracy(lrn_default 0.005)

ONNX_backend_node_test(lstm_batchwise)
ONNX_backend_node_test(lstm_defaults)
ONNX_backend_node_test(lstm_with_initial_bias)
ONNX_backend_node_test(lstm_with_peepholes)
local_node_test(lstm_all_outputs)
local_node_test(lstm_activations)
local_node_test(lstm_bidirectional)
local_node_test(lstm_clip)
local_node_test(lstm_intermediate_h)
local_node_test(lstm_missing_inputs)
local_node_test(lstm_reverse)
local_node_test(lstm_seq_length)
local_node_test(lstm_simple)
local_node_test(lstm_with_initial_state)
local_node_test(lstm_y_c)

ONNX_backend_node_test(matmul_2d)
#ONNX_backend_node_test(matmul_3d)
#ONNX_backend_node_test(matmul_4d)

ONNX_backend_node_test(matmulinteger)

ONNX_backend_node_test(max_example)
#ONNX_backend_node_test(max_float64)
ONNX_backend_node_test(max_int64)
ONNX_backend_node_test(max_two_inputs)
ONNX_backend_node_test(max_uint64)
#ONNX_backend_node_test(max_float16)
ONNX_backend_node_test(max_int16)
ONNX_backend_node_test(max_int8)
ONNX_backend_node_test(max_uint16)
ONNX_backend_node_test(max_uint8)
ONNX_backend_node_test(max_float32)
ONNX_backend_node_test(max_int32)
ONNX_backend_node_test(max_one_input)
ONNX_backend_node_test(max_uint32)

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

ONNX_backend_node_test(mean_example)
ONNX_backend_node_test(mean_one_input)
ONNX_backend_node_test(mean_two_inputs)

ONNX_backend_node_test(min_example)
#ONNX_backend_node_test(min_float64)
ONNX_backend_node_test(min_int64)
ONNX_backend_node_test(min_two_inputs)
ONNX_backend_node_test(min_uint64)
#ONNX_backend_node_test(min_float16)
ONNX_backend_node_test(min_int16)
ONNX_backend_node_test(min_int8)
ONNX_backend_node_test(min_uint16)
ONNX_backend_node_test(min_uint8)
ONNX_backend_node_test(min_float32)
ONNX_backend_node_test(min_int32)
ONNX_backend_node_test(min_one_input)
ONNX_backend_node_test(min_uint32)

# float64 not implemented in onnx2c
# mixed sign algorithm is not defined in onnx documentation
#ONNX_backend_node_test(mod_broadcast)
#ONNX_backend_node_test(mod_mixed_sign_float64)
#ONNX_backend_node_test(mod_mixed_sign_int8)
#ONNX_backend_node_test(mod_uint8)
ONNX_backend_node_test(mod_int64_fmod)
#ONNX_backend_node_test(mod_mixed_sign_int16)
#ONNX_backend_node_test(mod_uint16)
#ONNX_backend_node_test(mod_mixed_sign_float16)
#ONNX_backend_node_test(mod_mixed_sign_int32)
#ONNX_backend_node_test(mod_uint32)
ONNX_backend_node_test(mod_mixed_sign_float32)
#ONNX_backend_node_test(mod_mixed_sign_int64)
#ONNX_backend_node_test(mod_uint64)


ONNX_backend_node_test(mul)
ONNX_backend_node_test(mul_bcast)
ONNX_backend_node_test(mul_example)

ONNX_backend_node_test(neg)
ONNX_backend_node_test(neg_example)

ONNX_backend_node_test(not_2d)
ONNX_backend_node_test(not_3d)
ONNX_backend_node_test(not_4d)

ONNX_backend_node_test(pow)
ONNX_backend_node_test(pow_types_float)
ONNX_backend_node_test(pow_types_float32_uint64)
ONNX_backend_node_test(pow_types_int64_float32)
ONNX_backend_node_test(pow_bcast_array)
ONNX_backend_node_test(pow_types_float32_int32)
ONNX_backend_node_test(pow_types_int)
ONNX_backend_node_test(pow_types_int64_int64)
ONNX_backend_node_test(pow_bcast_scalar)
ONNX_backend_node_test(pow_types_float32_int64)
ONNX_backend_node_test(pow_types_int32_float32)
ONNX_backend_node_test(pow_example)
ONNX_backend_node_test(pow_types_float32_uint32)
ONNX_backend_node_test(pow_types_int32_int32)

ONNX_backend_node_test(prelu_broadcast)
ONNX_backend_node_test(prelu_example)

ONNX_backend_node_test(or2d)
ONNX_backend_node_test(or4d)
ONNX_backend_node_test(or_bcast3v2d)
ONNX_backend_node_test(or_bcast4v3d)
ONNX_backend_node_test(or3d)
ONNX_backend_node_test(or_bcast3v1d)
ONNX_backend_node_test(or_bcast4v2d)
ONNX_backend_node_test(or_bcast4v4d)

ONNX_backend_node_test(range_float_type_positive_delta)
ONNX_backend_node_test(range_int32_type_negative_delta)
# Not unit tests - needs Cast node
#ONNX_backend_node_test(range_float_type_positive_delta_expanded)
#ONNX_backend_node_test(range_int32_type_negative_delta_expanded)

ONNX_backend_node_test(reciprocal)
ONNX_backend_node_test(reciprocal_example)

ONNX_backend_node_test(relu)

ONNX_backend_node_test(reshape_extended_dims)
ONNX_backend_node_test(reshape_negative_extended_dims)
ONNX_backend_node_test(reshape_reduced_dims)
ONNX_backend_node_test(reshape_reordered_last_dims)
ONNX_backend_node_test(reshape_zero_dim)
ONNX_backend_node_test(reshape_negative_dim)
ONNX_backend_node_test(reshape_one_dim)
ONNX_backend_node_test(reshape_reordered_all_dims)
ONNX_backend_node_test(reshape_zero_and_negative_dim)

#ONNX_backend_node_test(resize_downsample_scales_cubic)
#ONNX_backend_node_test(resize_downsample_scales_cubic_align_corners)
#ONNX_backend_node_test(resize_downsample_scales_cubic_A_n0p5_exclude_outside)
ONNX_backend_node_test(resize_downsample_scales_linear)
# See issue https://github.com/onnx/onnx/issues/2462
#ONNX_backend_node_test(resize_downsample_scales_linear_align_corners)
ONNX_backend_node_test(resize_downsample_scales_nearest)
#ONNX_backend_node_test(resize_downsample_sizes_cubic)
ONNX_backend_node_test(resize_downsample_sizes_linear_pytorch_half_pixel)
ONNX_backend_node_test(resize_downsample_sizes_nearest)
ONNX_backend_node_test(resize_downsample_sizes_nearest_tf_half_pixel_for_nn)
ONNX_backend_node_test(resize_upsample_scales_linear_align_corners)
#ONNX_backend_node_test(resize_upsample_scales_cubic)
#ONNX_backend_node_test(resize_upsample_scales_cubic_asymmetric)
#ONNX_backend_node_test(resize_upsample_scales_cubic_align_corners)
#ONNX_backend_node_test(resize_upsample_scales_cubic_A_n0p5_exclude_outside)
ONNX_backend_node_test(resize_upsample_scales_linear)
ONNX_backend_node_test(resize_upsample_scales_nearest)
#ONNX_backend_node_test(resize_upsample_sizes_cubic)
ONNX_backend_node_test(resize_upsample_sizes_nearest)
ONNX_backend_node_test(resize_upsample_sizes_nearest_ceil_half_pixel)
ONNX_backend_node_test(resize_upsample_sizes_nearest_floor_align_corners)
ONNX_backend_node_test(resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric)
#ONNX_backend_node_test(resize_tf_crop_and_resize)
local_node_test(resize_downsample_sizes_linear_1D)
local_node_test(resize_downsample_sizes_linear_1D_align)

# Does not pass: ONNX rounding mode is towards even integer, which does not
# exist (I think) in C
#ONNX_backend_node_test(round)

ONNX_backend_node_test(selu)
ONNX_backend_node_test(selu_default)
ONNX_backend_node_test(selu_example)

ONNX_backend_node_test(sigmoid)
ONNX_backend_node_test(sigmoid_example)

ONNX_backend_node_test(sign)

ONNX_backend_node_test(sin)
ONNX_backend_node_test(sin_example)

ONNX_backend_node_test(sinh)
ONNX_backend_node_test(sinh_example)

ONNX_backend_node_test(shape)
ONNX_backend_node_test(shape_example)

ONNX_backend_node_test(slice)
ONNX_backend_node_test(slice_default_steps)
ONNX_backend_node_test(slice_neg)
ONNX_backend_node_test(slice_neg_steps)
ONNX_backend_node_test(slice_default_axes)
ONNX_backend_node_test(slice_end_out_of_bounds)
ONNX_backend_node_test(slice_negative_axes)
#ONNX_backend_node_test(slice_start_out_of_bounds)
local_node_test(slice_end_INT64_MAX)

ONNX_backend_node_test(softmax_axis_0)
ONNX_backend_node_test(softmax_axis_1)
ONNX_backend_node_test(softmax_axis_2)
ONNX_backend_node_test(softmax_default_axis)
ONNX_backend_node_test(softmax_example)
ONNX_backend_node_test(softmax_large_number)
ONNX_backend_node_test(softmax_negative_axis)

ONNX_backend_node_test(softplus)
ONNX_backend_node_test(softplus_example)

ONNX_backend_node_test(softsign)
ONNX_backend_node_test(softsign_example)

ONNX_backend_node_test(sqrt)
ONNX_backend_node_test(sqrt_example)

ONNX_backend_node_test(squeeze)
ONNX_backend_node_test(squeeze_negative_axes)

ONNX_backend_node_test(sub)
ONNX_backend_node_test(sub_bcast)
ONNX_backend_node_test(sub_example)
ONNX_backend_node_test(sub_uint8)

ONNX_backend_node_test(sum_example)
ONNX_backend_node_test(sum_one_input)
ONNX_backend_node_test(sum_two_inputs)

ONNX_backend_node_test(tan)
ONNX_backend_node_test(tan_example)

ONNX_backend_node_test(tanh)
ONNX_backend_node_test(tanh_example)

ONNX_backend_node_test(transpose_default)
ONNX_backend_node_test(transpose_all_permutations_0)
ONNX_backend_node_test(transpose_all_permutations_1)
ONNX_backend_node_test(transpose_all_permutations_2)
ONNX_backend_node_test(transpose_all_permutations_3)
ONNX_backend_node_test(transpose_all_permutations_4)
ONNX_backend_node_test(transpose_all_permutations_5)

ONNX_backend_node_test(thresholdedrelu)
ONNX_backend_node_test(thresholdedrelu_default)
ONNX_backend_node_test(thresholdedrelu_example)

ONNX_backend_node_test(unsqueeze_axis_0)
ONNX_backend_node_test(unsqueeze_axis_2)
ONNX_backend_node_test(unsqueeze_negative_axes)
ONNX_backend_node_test(unsqueeze_two_axes)
ONNX_backend_node_test(unsqueeze_axis_1)
ONNX_backend_node_test(unsqueeze_axis_3)
ONNX_backend_node_test(unsqueeze_three_axes)
ONNX_backend_node_test(unsqueeze_unsorted_axes)

ONNX_backend_node_test(xor2d)
ONNX_backend_node_test(xor4d)
ONNX_backend_node_test(xor_bcast3v2d)
ONNX_backend_node_test(xor_bcast4v3d)
ONNX_backend_node_test(xor3d)
ONNX_backend_node_test(xor_bcast3v1d)
ONNX_backend_node_test(xor_bcast4v2d)
ONNX_backend_node_test(xor_bcast4v4d)



add_subdirectory(old_onnx_backend)

# More end-to-end kind of tests here
add_subdirectory(tfl_helloworld)
add_subdirectory(mnist)
add_subdirectory(velardo)
add_subdirectory(simple_networks)
add_subdirectory(onnx_model_zoo)

# Misc. onnx2c unit tests
local_node_test(matmul_precision)
local_node_test(nodes_out_of_order)

