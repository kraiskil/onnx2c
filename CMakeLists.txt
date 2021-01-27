cmake_minimum_required(VERSION 3.13)
# 3.13 because of target_link_options
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

project(onnx2c
	VERSION 0.0.1
	LANGUAGES C CXX
	) 
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror")

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(src)

add_custom_command(
	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.h
	COMMAND protoc --proto_path=${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/ --cpp_out=. ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/onnx.proto
	MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/onnx/onnx/onnx.proto
	)


add_library(onnx2c_lib STATIC
	src/graph.cc
	src/graph_print.cc
	src/node.cc
	src/tensor.cc
	src/util.cc
	${CMAKE_CURRENT_BINARY_DIR}/onnx.pb.cc
)
target_compile_options(onnx2c_lib
	PUBLIC
		-I${CMAKE_CURRENT_BINARY_DIR}
	PRIVATE
		-Isrc
		-I${CMAKE_CURRENT_SOURCE_DIR}/aixlog/include
		-Wall
	)
add_executable( onnx2c
	src/main.cc
	)
target_compile_options(onnx2c
	PRIVATE
		-Isrc
		-I${CMAKE_CURRENT_SOURCE_DIR}/aixlog/include
		-Wall
	)
target_link_libraries(onnx2c onnx2c_lib ${Protobuf_LIBRARIES})

enable_testing()
add_subdirectory(test)

