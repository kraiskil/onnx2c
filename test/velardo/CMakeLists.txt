add_custom_command(OUTPUT lesson14_generated.c
	COMMAND onnx2c ${CMAKE_CURRENT_SOURCE_DIR}/lesson14.onnx > lesson14_generated.c
	DEPENDS 
		${CMAKE_CURRENT_SOURCE_DIR}/lesson14.onnx) 
add_executable(lesson_14 lesson14_generated.c main.c)
target_compile_options(lesson_14
	PRIVATE -DLESSON_14)
target_link_libraries(lesson_14 onnx2c_lib protobuf)

add_test(Velardo_lesson14 lesson_14)


