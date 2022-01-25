/* This file is part of onnx2c.
 *
 * Pad node.
 */
#include "node.h"
namespace toC {

class Pad : public Node {
	public:
	Pad() {
		op_name = "Pad";
		data=output=pads_tensor=constant_value=0;
		value_attribute=0;
		mode = "constant";
	}

	/* attributes */
	std::string mode;
	/* These two removed in version 11 */
	std::vector<int64_t> pads_attribute;
	float value_attribute;

	// input and output tensors
	const Tensor *data;
	const Tensor *output;
	// inputs added in version 11
	const Tensor *pads_tensor;
	const Tensor *constant_value;


	// The actual paddings used. Collected from pads_attribute or pads_tensor
	std::vector<int64_t> paddings_start;
	std::vector<int64_t> paddings_end;
	float constant; //ditto

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override;
	virtual void print(std::ostream &dst) const override;
};
}

