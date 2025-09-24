function( ONNX_old_backend_node_test onnx_version test_name)
	ONNX_type_test(
		${onnx_version}_${test_name}
		${CMAKE_CURRENT_SOURCE_DIR}/${onnx_version}/${test_name}
		ONNX_backend_${onnx_version}_${test_name}
		0.00002
		0
	)
endfunction()

function( ONNX_old_backend_node_test_singlefile onnx_version test_name)
	ONNXtype_test_singlefile(
		${onnx_version}_${test_name}
		${CMAKE_CURRENT_SOURCE_DIR}/${onnx_version}/${test_name}
		ONNX_backend_${onnx_version}_${test_name}
		0.00002
		0
	)
endfunction()

ONNX_old_backend_node_test_singlefile(5 test_resize_downsample_sizes_nearest_tf_half_pixel_for_nn)

ONNX_old_backend_node_test_singlefile(7 test_split_variable_parts_1d)
ONNX_old_backend_node_test_singlefile(7 test_split_variable_parts_2d)
ONNX_old_backend_node_test_singlefile(7 test_split_variable_parts_default_axis)

ONNX_old_backend_node_test(8 test_convtranspose_with_kernel)

ONNX_old_backend_node_test(9 test_slice)
ONNX_old_backend_node_test(9 test_slice_default_axes)
ONNX_old_backend_node_test(9 test_slice_end_out_of_bounds)
ONNX_old_backend_node_test(9 test_slice_neg)

ONNX_old_backend_node_test(11 test_softmax_axis_0)
ONNX_old_backend_node_test(11 test_softmax_axis_1)
ONNX_old_backend_node_test(11 test_softmax_default_axis)
ONNX_old_backend_node_test(11 test_squeeze)
ONNX_old_backend_node_test(11 test_squeeze_negative_axes)
ONNX_old_backend_node_test(11 test_unsqueeze_axis_0)
ONNX_old_backend_node_test(11 test_unsqueeze_axis_1)
ONNX_old_backend_node_test(11 test_unsqueeze_axis_2)
ONNX_old_backend_node_test(11 test_unsqueeze_axis_3)
ONNX_old_backend_node_test(11 test_unsqueeze_negative_axes)
ONNX_old_backend_node_test(11 test_unsqueeze_three_axes)
ONNX_old_backend_node_test(11 test_unsqueeze_two_axes)
ONNX_old_backend_node_test(11 test_unsqueeze_unsorted_axes)

ONNX_old_backend_node_test(12 test_pow_types_float)
ONNX_old_backend_node_test(12 test_pow_types_int)

