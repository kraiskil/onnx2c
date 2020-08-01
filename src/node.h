#pragma once
#include <string>
#include "onnx.pb.h"
#include "op.h"

namespace toC {

class Tensor;

/* The ONNX node, or computation kernel */
class Node {
	public:
	bool isResolved;
	std::string name;
	std::string op_name;
	const Op *op;

	std::vector<const Tensor*> inputs;
	std::vector<const Tensor*> outputs;
};
}
