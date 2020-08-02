enable_testing()

add_custom_command(
	OUTPUT
	generated.c
	COMMAND
	onnx2c ${CMAKE_CURRENT_SOURCE_DIR}/model.onnx > generated.c
	DEPENDS
	onnx2c model.onnx
	)
add_executable(test_tfl_helloworld
	test.c
	generated.c
	)
target_link_libraries(test_tfl_helloworld m)

add_test(
	NAME tfl_helloworld
	COMMAND test_tfl_helloworld
	)
