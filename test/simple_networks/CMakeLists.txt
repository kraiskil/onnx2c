
function( simple_test test_name output_size)
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main.c)
	target_link_libraries(${test_name} onnx2c_lib protobuf)
	target_compile_options(${test_name}
		PRIVATE
			-DOUTPUT_SIZE=${output_size})
	add_test(small_${test_name} ${test_name})
endfunction()


simple_test( conv_k2 16 )
simple_test( conv_k2_maxpool_k2 4 )
simple_test( conv_k2_s2 4 )
simple_test( maxpool_k2 16 )
simple_test( maxpool_k2_s2 4 )
