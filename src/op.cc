
#include "error.h"
#include "graph.h"
#include "op.h"


using namespace toC;

bool Op::typeConstraint_highPrecisionNumeric(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
		|| t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}

bool Op::typeConstraint_floatingPoints(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}

#include "nodes/add.h"
#include "nodes/matmul.h"
#include "nodes/relu.h"

void Graph::initializeOpArray(void)
{
	ops.push_back(new Add);
	ops.push_back(new MatMul);
	ops.push_back(new Relu);
}
