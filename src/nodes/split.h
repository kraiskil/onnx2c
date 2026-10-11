/* This file is part of onnx2c.
 *
 * Split.
 * "Split a tensor into a list of tensors, along the specified ‘axis’."
 */

#include <numeric>

namespace toC {

class Split : public Node {
	public:
	Split()
	{
		op_name = "Split";
	}

	int64_t axis = 0;
	// 'num_outputs' attribute (ONNX opset 18). 0 == not specified.
	int64_t num_outputs_attr = 0;
	// Sizes of the split parts, filled in resolve().
	std::vector<int64_t> split_sizes;
	// True when the sizes above come from the 'split' input tensor.
	bool split_from_input = false;

	virtual void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			if (a.name() == "axis")
				axis = parse_attribute_int(a);
			else if (a.name() == "num_outputs")
				num_outputs_attr = parse_attribute_int(a);
			else if (a.name() == "split")
				ERROR("Attribute " << a.name() << " deprecated and not supported");
			else
				ERROR("Bad attribute " << a.name() << " to split");
		}
	}

	virtual void print(std::ostream& dst) const override
	{
		const Tensor* input = get_input_tensor(0);
		int64_t num_outputs = split_sizes.size();
		int64_t num_dims = input->data_dim.size();
		auto io_type_str = input->data_type_str();

		INDT_1 "/*" << op_name << "*/" << std::endl;
		INDT_1 "const size_t axis = " << std::to_string(axis) << ";" << std::endl;

		// When the split sizes do not come from the 'split' input tensor,
		// materialize them as a local array (the loop below indexes 'split').
		if (!split_from_input) {
			INDT_1 "const int64_t split[" << num_outputs << "] = {";
			for (int64_t i = 0; i < num_outputs; i++) {
				dst << split_sizes[i];
				if (i < num_outputs - 1)
					dst << ", ";
			}
			dst << "};" << std::endl;
		}
		INDT_1 "const size_t dims[" << std::to_string(num_dims) << "] = {";

		for (int64_t i = 0; i < num_dims; i++) {
			int64_t dim = 1;
			for (int64_t j = i + 1; j < num_dims; j++) {
				dim *= input->data_dim[j];
			}
			dst << std::to_string(dim);

			if (i < num_dims - 1) {
				dst << ", ";
			}
		}

		dst << "};" << std::endl;
		INDT_1 << "size_t idx[" << std::to_string(num_dims) << "];" << std::endl
		       << std::endl;

		INDT_1 << "for (size_t i = 0; i < (";
		for (int64_t i = 0; i < num_dims; i++) {
			dst << std::to_string(input->data_dim[i]);
			if (i < num_dims - 1) {
				dst << " * ";
			}
		}

		dst << "); i++)" << std::endl;

		INDT_1 << "{" << std::endl;
		INDT_2 << "size_t t = i;" << std::endl
		       << std::endl;

		INDT_2 << "for (size_t j = 0; j < " << std::to_string(num_dims) << "; j++)" << std::endl;
		INDT_2 << "{" << std::endl;
		INDT_3 << "idx[j] = t / dims[j];" << std::endl;
		INDT_3 << "t %= dims[j];" << std::endl;

		INDT_2 << "}" << std::endl
		       << std::endl;
		INDT_2 << "" << io_type_str << " x = ((" << io_type_str << " *)input)[i];" << std::endl;

		INDT_2 << "size_t split_idx = idx[axis];" << std::endl;
		INDT_2 << "size_t split_sum = 0;" << std::endl;
		INDT_2 << "size_t out_idx;" << std::endl
		       << std::endl;

		INDT_2 << "int64_t offset = 0;" << std::endl;
		INDT_2 << "for (out_idx = 0; out_idx < " << num_outputs << "; out_idx++)" << std::endl;
		INDT_2 << "{" << std::endl;
		INDT_3 << "split_sum += split[out_idx];" << std::endl;
		INDT_3 << "if (split_idx < split_sum)" << std::endl;
		INDT_3 << "{" << std::endl;
		INDT_4 << "break;" << std::endl;
		INDT_3 << "}" << std::endl;
		INDT_3 << "offset += split[out_idx];" << std::endl;
		INDT_2 << "}" << std::endl
		       << std::endl;

		INDT_2 << "switch (out_idx)" << std::endl;
		INDT_2 << "{" << std::endl;

		for (int64_t i = 0; i < num_outputs; i++) {
			INDT_3 << "case " << std::to_string(i) << ":" << std::endl;
			INDT_4 << "output_" << std::to_string(i);
			for (int64_t j = 0; j < num_dims; j++) {
				dst << "[";
				if (j == axis) {
					dst << "idx[" << std::to_string(j) << "] - offset";
				}
				else {
					dst << "idx[" << std::to_string(j) << "]";
				}
				dst << "]";
			}
			dst << " = x;" << std::endl;
			INDT_4 << "break;" << std::endl
			       << std::endl;
		}

		INDT_3 << "default:" << std::endl;
		INDT_4 << "break;" << std::endl;

		INDT_2 << "}" << std::endl
		       << std::endl;

		INDT_1 << "}" << std::endl;

		dst << std::endl;
	}

	virtual void resolve(void) override
	{
		const Tensor* input = get_input_tensor(0);

		if (axis < 0) {
			axis = input->rank() + axis;
		}

		name_input(0, "input");

		unsigned num_inputs = get_number_of_inputs();

		if (num_inputs >= 2) {
			// Opset 13+: the split sizes may be given as an input tensor.
			const Tensor* split = get_input_tensor(1);

			if (!split->isConst)
				ERROR("Only constant split input in Split nodes supported");

			name_input(1, "split");
			split_from_input = true;

			int64_t split_sum = 0;
			for (int i = 0; i < split->data_num_elem(); i++) {
				auto e = split->get_data_element(i);
				if (e < 0) {
					ERROR("'split' values must be greater than zero");
				}
				split_sizes.push_back(e);
				split_sum += e;
			}

			if (input->data_dim[axis] != split_sum) {
				ERROR("Sum of 'split' values must be equal to the dim value at 'axis' parameter ("
				      << input->data_dim[axis] << ")");
			}
		}
		else {
			// Opset 18: split into equal parts. The number of parts is given
			// by the 'num_outputs' attribute or, when it is absent, by the
			// number of outputs the node declares in the graph.
			int64_t n = num_outputs_attr;
			if (n <= 0)
				n = (int64_t)get_number_of_declared_outputs();
			if (n <= 0)
				ERROR("Split node without a 'split' input or a 'num_outputs' attribute");

			int64_t dim = input->data_dim[axis];

			if (dim % n == 0) {
				for (int64_t i = 0; i < n; i++)
					split_sizes.push_back(dim / n);
			}
			else {
				// Follow the ONNX reference implementation: each part gets
				// floor(dim/n)+1 elements and the last part absorbs the
				// remainder, making it the smaller one.
				int64_t chunk = dim / n + 1;
				for (int64_t i = 0; i < n; i++)
					split_sizes.push_back(chunk);
				split_sizes[n - 1] += dim - chunk * n;
			}
		}

		for (size_t i = 0; i < split_sizes.size(); i++) {
			Tensor* rv = new Tensor;

			rv->data_type = input->data_type;
			for (uint64_t j = 0; j < input->data_dim.size(); j++) {
				if (j == (uint64_t)axis) {
					rv->data_dim.push_back(split_sizes[i]);
				}
				else {
					rv->data_dim.push_back(input->data_dim[j]);
				}
			}
			register_output(rv, "output_" + std::to_string(i));
		}
	}
};
} // namespace toC
