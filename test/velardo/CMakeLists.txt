compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/lesson9.onnx lesson9_generated.c )
add_executable(lesson_9 lesson9_generated.c main_lesson9.c)
target_link_libraries(lesson_9 onnx2c_lib protobuf)
add_test(Velardo_lesson9 lesson_9)


compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/lesson14.onnx lesson14_generated.c )
add_executable(lesson_14 lesson14_generated.c main.c)
target_compile_options(lesson_14
	PRIVATE -DLESSON_14)
target_link_libraries(lesson_14 onnx2c_lib protobuf)
add_test(Velardo_lesson14 lesson_14)


