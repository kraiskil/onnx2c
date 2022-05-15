/* This file is part of onnx2c.
 *
 * ScatterND node.
 */
#include "node.h"

namespace toC {

class ScatterND : public Node {
	public:
	ScatterND() {
		op_name = "ScatterND";
		data=indices=updates=output=NULL;
		reduction="";
	}
	std::string reduction;

	const Tensor *data;
	const Tensor *indices;
	const Tensor *updates;
	const Tensor *output;

	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};

}

