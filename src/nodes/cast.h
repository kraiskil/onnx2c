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
		to=-1;
	}
	int to;

	std::string output_type;

	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};
} // namespace

