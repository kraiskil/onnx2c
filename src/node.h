#pragma once
#include <string>
#include "error.h"
#include "onnx.pb.h"
#include "util.h"

namespace toC {

class Tensor;

/* The ONNX node, or computation kernel. *
 * Node is a virtual parent class for each of the
 * ONNX node "types" or "operands" (e.g. Add, Relu, ...)
 * 
 */
class Node {
	public:
	bool isResolved;
	std::string onnx_name; //ONNX name of the individual node
	std::string op_name;   //ONNX name of node type
	std::vector<const Tensor*> inputs;
	std::vector<const Tensor*> outputs;

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name since ONNX allows tensors and nodes
	 * to have the same name */
	std::string c_name(void) const
	{
		return "node_" + cify_name(onnx_name);
	}


	/* Print the C implmementation of the operator */
	virtual void print(std::ostream &destination) const = 0; 

	/* Figure out in what format the output is in.
	 * Return values are pointers to Tensor values, allocated with new. Ownership given to caller.
	 * This function may not fail: call only with all inputs given & resolved */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) = 0;

	/* Not all node types have attributes. Override where needed */
	virtual void parseAttributes( onnx::NodeProto &node )
	{
		ERROR("Attribute parsing not implemented for node operation type " << op_name);
	}

	/* TODO: these should be part of class Tensor... */
	/* Check input constraints, as used in 
	 * https://github.com/onnx/onnx/blob/master/docs/Operators.md
	 */
	/* (u)int32, (u)int64, float16/32/64, bfloat*/
	bool typeConstraint_highPrecisionNumeric(const Tensor *t) const;
	/* float16/32/64, bfloat */
	bool typeConstraint_allFloatingPoints(const Tensor *t) const;
	/* float16/32/64, (not bfloat!) */
	bool typeConstraint_plainFloatingPoints(const Tensor *t) const;
	bool typeConstraint_int64(const Tensor *t) const;
	/* int8 or uint8 */
	bool typeConstraint_8bit(const Tensor *t) const;

	/* Do Multidirectional Broadcasting dimension extensions:
	 * https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
	 */
	void multidirectional_broadcast_size(
		const std::vector<int> A,
		const std::vector<int> B,
		std::vector<int> &result) const;
};
}
