/* This file is part of onnx2c.
 *
 * Expand node.
 */
#include "node.h"

namespace toC {

class Expand : public Node {
	public:
	Expand() {
		op_name = "Expand";
		input=shape=output=NULL;
	}

	const Tensor *input;
	const Tensor *shape;
	const Tensor *output;

	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;


	std::vector<int32_t> resolve_output_shape(void) const;
};

} // namespace

