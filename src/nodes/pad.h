/* This file is part of onnx2c.
 *
 * Pad node.
 */
#include "node.h"
namespace toC {

class Pad : public Node {
	public:
	Pad()
	{
		op_name = "Pad";
		value_attribute = 0;
		mode = "constant";
	}

	/* attributes */
	std::string mode;
	/* These two removed in version 11 */
	std::vector<int64_t> pads_attribute;
	float value_attribute;

	// The actual paddings used. Collected from pads_attribute or pads_tensor
	std::vector<int64_t> paddings_start;
	std::vector<int64_t> paddings_end;
	float constant; //	ditto

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};
} // namespace toC
