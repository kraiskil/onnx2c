# im2col optimization pass tests

function(add_im2col_test model_name)
    set(onnx_file ${CMAKE_CURRENT_SOURCE_DIR}/${model_name}.onnx)
    set(generated_c ${CMAKE_CURRENT_BINARY_DIR}/${model_name}_generated.c)

    add_custom_command(
        OUTPUT ${generated_c}
        COMMAND ${CMAKE_BINARY_DIR}/onnx2c -p im2col -l 0 ${onnx_file} > ${generated_c}
        DEPENDS onnx2c ${onnx_file}
        COMMENT "Generating im2col C code for ${model_name}"
        VERBATIM
    )

    add_executable(im2col_${model_name}
        ${generated_c}
        ${CMAKE_CURRENT_SOURCE_DIR}/test_runner.c
    )

    target_include_directories(im2col_${model_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_options(im2col_${model_name} PRIVATE -Wno-error)

    target_compile_definitions(im2col_${model_name} PRIVATE
        TEST_${model_name}=1
    )

    target_link_libraries(im2col_${model_name} m)

    add_test(NAME im2col_${model_name} COMMAND im2col_${model_name})
endfunction()

add_im2col_test(conv_simple)
add_im2col_test(conv_stride2)
add_im2col_test(depthwise_conv)
add_im2col_test(pointwise_conv)
add_im2col_test(multi_channel_conv)