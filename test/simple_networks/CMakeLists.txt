# Small networks generated with the accompanying
# python scripts, with different parameters.
#
# To add a new one might need a kludge, but general guidlines are
# -create new TEST.onnx model with demo.py
# -input is fixed to the matrix in that script
# -copy its output into TEST.c
# -add new simple_test() line.


function( simple_test test_name output_size num_chan)
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main.c)
	target_link_libraries(${test_name} onnx2c_lib ${Protobuf_LIBRARIES})
	target_compile_options(${test_name}
		PRIVATE
			-Wall -Werror
			-DOUTPUT_SIZE=${output_size}
			-DNUM_CHAN=${num_chan})
	add_test(small_${test_name} ${test_name})
endfunction()

simple_test( conv_k2 16 1 )
simple_test( conv_k2_maxpool_k2 4 1 )
simple_test( conv_k2_s2 4 1 )
simple_test( conv_3ch 16 3 )
simple_test( conv_2kernels 32 1 )
simple_test( conv_2kernels_randombias 32 1 )
simple_test( conv_2ch_3kernels_randombias 48 2)
simple_test( maxpool_k2 16 1 )
simple_test( maxpool_k2_s2 4 1 )


# LSTM being a recursive model, several invocations might be needed (the 'r' parameter in the test name).
# This a kludge left in from debugging the initial LSTM implementation, but there is no point removing it
# now. This is not a fragile test - should not break.
function( lstm_test test_name)
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main_lstm.c)
	target_link_libraries(${test_name} onnx2c_lib ${Protobuf_LIBRARIES})
	target_compile_options(${test_name}
		PRIVATE
			-Wall -Werror
			-DOUTPUT_SIZE=${output_size})
	add_test(small_${test_name} ${test_name})
endfunction()
lstm_test( lstm_k1_b1_r1 )
lstm_test( lstm_k1_b1_r1_relu )

function( simple_test_treeensembleclassifier test_name output_size num_chan)
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main_treeensembleclassifier.c)
	target_link_libraries(${test_name} onnx2c_lib ${Protobuf_LIBRARIES})
	target_compile_options(${test_name}
		PRIVATE
			-Wall -Werror
			-DOUTPUT_SIZE=${output_size}
			-DNUM_CHAN=${num_chan})
	add_test(small_${test_name} ${test_name})
endfunction()

simple_test_treeensembleclassifier( treeensembleclassifier 3 1 )

function( simple_test_scalar test_name )
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main_scalar.c)
	target_link_libraries(${test_name} onnx2c_lib ${Protobuf_LIBRARIES})
	target_compile_options(${test_name}
		PRIVATE
			-Wall -Werror)
	add_test(small_${test_name} ${test_name})
endfunction()

simple_test_scalar( scalar_identity )
simple_test_scalar( scalar_abs )
simple_test_scalar( scalar_add )

compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/random_uniform.onnx random_uniform_generated.c)
add_executable(random_uniform random_uniform_generated.c main_random_uniform.c)
target_link_libraries(random_uniform onnx2c_lib ${Protobuf_LIBRARIES})
target_compile_options(random_uniform
	PRIVATE
		-Wall -Werror)
add_test(small_random_uniform random_uniform)

function( simple_test_fp test_name )
	compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/${test_name}.onnx ${test_name}_generated.c)
	add_executable(${test_name} ${test_name}_generated.c ${test_name}.c  main_fp.c)
	target_link_libraries(${test_name} onnx2c_lib ${Protobuf_LIBRARIES})
	target_compile_options(${test_name}
		PRIVATE
			-Wall -Werror)
	add_test(small_${test_name} ${test_name})
endfunction()

simple_test_fp( fp_float16 )
simple_test_fp( fp_bfloat16 )
