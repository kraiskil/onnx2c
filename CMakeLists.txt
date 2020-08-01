cmake_minimum_required(VERSION 3.13)
# 3.13 because of target_link_options
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

project(onnx2c
	VERSION 0.0.1
	LANGUAGES CXX
	) 
set (CMAKE_CXX_STANDARD 11)


add_executable( onnx2c 
	src/graph.cc
	src/graph_print.cc
	src/main.cc
	src/onnx.pb.cc
	src/op.cc
	)
target_compile_options(onnx2c
	PRIVATE
		-Isrc
		-Wall
	)
target_link_libraries(onnx2c protobuf)
