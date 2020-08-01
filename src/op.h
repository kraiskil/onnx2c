#pragma once
#include <string>
#include "node.h"
#include "onnx.pb.h"
#include "tensor.h"

namespace toC {
class Node;

/* Interface class to ONNX operator implementations in toC */
class Op {
	public:
	std::string name;

	/* Print the C implmementation of the operator */
	virtual void print(std::ostream &destination, const Node *n) const = 0; 

	/* Figure out in what format the output is in.
	 * Return values are pointers to Tensor values, allocated with new. Ownership given to caller.
	 * This function may not fail: call only with all inputs given & resolved */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) const = 0;


	/* Check input constraints, as used in 
	 * https://github.com/onnx/onnx/blob/master/docs/Operators.md
	 */
	/* (u)int32, (u)int64, float16/32/64, bfloat*/
	bool typeConstraint_highPrecisionNumeric(const Tensor *t) const;
	/* float16/32/64, bfloat */
	bool typeConstraint_floatingPoints(const Tensor *t) const;
};
}

