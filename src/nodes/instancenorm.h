/* This file is part of onnx2c.
 *
 * InstanceNormalization node.
 * When implementing a new node, use this template
 * as a starting point.
 * Replace all occurances of InstanceNormalization in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */
#pragma once
#include "node.h"
namespace toC {

class InstanceNormalization : public Node {
	public:
	InstanceNormalization() {
		op_name = "InstanceNormalization";
		input=scale=B=output=NULL;
		epsilon = 1e-5;
	}
	/* Node attributes */
	float epsilon;

	// input and output
	const Tensor *input;
	const Tensor *scale;
	const Tensor *B;
	const Tensor *output;

	virtual void print(std::ostream &dst) const override;
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override;
	virtual void parseAttributes( onnx::NodeProto &node ) override;
};
}

