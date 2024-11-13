/* This file is part of onnx2c.
 *
 * ReduceMean node.
 *
 */
#include "node.h"
#include "util.h"

namespace toC {

class ReduceMean : public Node {
	public:
	ReduceMean() {
		op_name = "ReduceMean";
	}

	constexpr static const char* kInputName = "input";
	constexpr static const char* kOutputName = "reduced";

	/* Attributes */
	std::vector<int64_t> axes;
	int keepdims;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
	private:
	void printLocationArray(std::ostream &dst, unsigned indent, int axis, int dims, const char* flatIndexVariable) const;
};

} // namespace

