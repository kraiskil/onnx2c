
file( GLOB res
	LIST_DIRECTORIES true
	${CMAKE_CURRENT_SOURCE_DIR}/*/* )

function( ONNX_old_backend_node_test test_dir)
	# Get test name (== last directory in path) and ONNX version directory
	get_filename_component(test_name ${test_dir} NAME)
	get_filename_component(full_path ${test_dir} DIRECTORY)
	get_filename_component(onnx_version ${full_path} NAME)

	ONNX_type_test(
		${onnx_version}_${test_name}
		${CMAKE_CURRENT_SOURCE_DIR}/${onnx_version}/${test_name}
		ONNX_backend_${onnx_version}_${test_name}
		0.00002
		0
	)
endfunction()

foreach( test ${res})
	ONNX_old_backend_node_test(${test})
endforeach()
