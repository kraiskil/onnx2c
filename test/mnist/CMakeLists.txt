
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist0 0.01 0)
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist1 0.01 1)
ONNX_type_test(mnist ${CMAKE_CURRENT_SOURCE_DIR} mnist2 0.01 2)
compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/model.onnx mnist_generated.c )
add_executable(mnist_static test.cc mnist_generated.c)
target_link_libraries(mnist_static onnx2c_lib ${Protobuf_LIBRARIES})
add_test(mnist_static mnist_static)

compile_onnx( ${CMAKE_CURRENT_SOURCE_DIR}/pytorch.onnx pytorch_generated.c )
add_executable(pytorch_mnist test_pytorch.cc pytorch_generated.c)
add_test(pytorch_mnist pytorch_mnist)
