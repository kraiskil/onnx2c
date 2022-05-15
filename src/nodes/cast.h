/* This file is part of onnx2c.
 *
 * Cast node. Casts between float/double/string/half floats.
 */
#include "node.h"

namespace toC {

class Cast : public Node {
	public:
	Cast() {
		op_name = "Cast";
		input=output=NULL;
		to=-1;
	}
	int to;

	std::string output_type;

	// input and output tensors
	const Tensor *input;
	const Tensor *output;

	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};
} // namespace

