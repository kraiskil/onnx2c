# Run ./download.sh in this directory.
# Disable any of the downloaded tests by removing the respective unpacked directory
# before running cmake.

#ONNX_type_test(node_name   data_dir                                 test cmake name    accuracy test_data_set)
if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tiny_yolov2 )
ONNX_type_test( tiny_yolov2 ${CMAKE_CURRENT_SOURCE_DIR}/tiny_yolov2  modelzoo_tinyyolo2_0 0.002  0)
ONNX_type_test( tiny_yolov2 ${CMAKE_CURRENT_SOURCE_DIR}/tiny_yolov2  modelzoo_tinyyolo2_1 0.002  1)
ONNX_type_test( tiny_yolov2 ${CMAKE_CURRENT_SOURCE_DIR}/tiny_yolov2  modelzoo_tinyyolo2_2 0.002  2)
endif()

if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_0   0.00001    0)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_1   0.00001    1)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_2   0.00001    2)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_3   0.00001    3)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_4   0.00001    4)
ONNX_type_test( alexnet     ${CMAKE_CURRENT_SOURCE_DIR}/bvlc_alexnet modelzoo_alexnet_5   0.00001    5)
endif()

if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_0    0.0001  0)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_1    0.0001  1)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_2    0.0001  2)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_3    0.0001  3)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_4    0.0001  4)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_5    0.0001  5)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_6    0.0001  6)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_7    0.0001  7)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_8    0.0001  8)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_9    0.0001  9)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_10   0.0001 10)
ONNX_type_test( squeezenet  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet   squeezenet_1.0_11   0.0001 11)
endif()

if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet1.1)
ONNX_type_test( squeezenet11  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet1.1 squeezenet_1.1_0  0.001  0)
ONNX_type_test( squeezenet11  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet1.1 squeezenet_1.1_1  0.001  1)
ONNX_type_test( squeezenet11  ${CMAKE_CURRENT_SOURCE_DIR}/squeezenet1.1 squeezenet_1.1_2  0.001  2)
endif()


