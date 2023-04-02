/* This file is part of onnx2c.
 *
 * ConstantOfShape node.
 * When implementing a new node, use this template
 * as a starting point.
 *
 * This file can be kept as a single .h file with an
 * in-header implementation, or it can be split into
 * a .h and a .cc file.
 *
 * Replace all occurances of ConstantOfShape in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */
#include "node.h"

namespace toC {

class ConstantOfShape : public Node {
	public:
	ConstantOfShape() {
		op_name = "ConstantOfShape";
		value=NULL;
	}

	// Attribute, not input
	const Tensor *value;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};
} // namespace

