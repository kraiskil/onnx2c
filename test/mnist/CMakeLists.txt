
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist0 0.01 0)
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist1 0.01 1)
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist2 0.01 2)
add_custom_command(OUTPUT mnist_generated.c
	COMMAND onnx2c ${CMAKE_CURRENT_SOURCE_DIR}/model.onnx > mnist_generated.c)
add_executable(mnist_static test.cc mnist_generated.c)
target_link_libraries(mnist_static onnx2c_lib protobuf)
add_test(mnist_static mnist_static)

