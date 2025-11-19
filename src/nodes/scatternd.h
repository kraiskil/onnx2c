/* This file is part of onnx2c.
 *
 * ScatterND node.
 */
#include "node.h"

namespace toC {

class ScatterND : public Node {
	public:
	ScatterND()
	{
		op_name = "ScatterND";
		reduction = "";
	}
	std::string reduction;

	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

} // namespace toC
