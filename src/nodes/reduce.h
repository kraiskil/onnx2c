/* This file is part of onnx2c.
 *
 * Reduce Operations
 */
#include "node.h"

namespace toC {

class Reduce : public Node {
	public:

	// Each instance of this class should override this lambda with the operation of the node type.
	std::function<const std::string (const std::string&, const std::string&)> elemet_operation =
		[](const std::string &a, const std::string &b){ ERROR("onnx2c internal error"); return ""; };


	Reduce(std::string op) {
		op_name = "Reduce" + op;

	}
	std::vector<int64_t> axes = {};		// can be negative
	std::vector<size_t> norm_axes= {};	// will be constructed to be between 0 and axis-size
	bool keepdims = 1;
	std::string initial_value;

	const Tensor *input;
	const Tensor *output;

	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;

	/* 	Function that calculates the number of elements that will be reduced 
		based on norm_axes and input shape*/
	int get_number_of_reduced_elements(void) const;
	/* Function that prints iterator over the output tensor for initilaization purpose*/
	std::string print_and_return_o_iterator(std::ostream &dst) const;
	/* Function that prints iteraor and returns two index strings that can be used to index the I/O tensors */
	std::pair<std::string, std::string> print_and_return_io_iterator(std::ostream &dst) const;
	/* Function that returns normalized axes as the axes argument can contain negative values */
	std::vector<size_t> normalized_axes(const Tensor *t) const;
};

}