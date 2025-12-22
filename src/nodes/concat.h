/* This file is part of onnx2c.
 *
 * Concate ... concatenates a number of input tensors
 * across a given dimension.
 */

namespace toC {

class Concat : public Node {
	public:
	Concat()
	{
		op_name = "Concat";
		axis = 1;
	}

	// attribute
	int axis;

	void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			if (a.name() == "axis") {
				axis = parse_attribute_int(a);
			}
			else {
				ERROR("Unknown attribute " << a.name());
			}
		}
	}

	void print(std::ostream& dst) const override
	{
		dst << "\t/* Concat */" << std::endl;

		size_t offset = 0;
		for (size_t i = 0; i < get_number_of_inputs(); i++) {
			const Tensor* input_tensor = get_input_tensor(i);

			std::string input_idx = "input_" + std::to_string(i);
			std::string result_idx = "concat_result";
			for (size_t j = 0; j < input_tensor->rank(); j++) {
				std::string lv = "i" + std::to_string(j);
				INDT_1 << "for (size_t " << lv << " = 0; ";
				dst << lv << "<" << input_tensor->data_dim[j] << "; ";
				dst << lv << "++)" << std::endl;

				input_idx += "[" + lv + "]";
				if (j == (size_t)axis) {
					result_idx += "[" + std::to_string(offset) + " + " + lv + "]";
				}
				else {
					result_idx += "[" + lv + "]";
				}
			}

			INDT_1 << "{" << std::endl;
			INDT_2 << result_idx << " = " << input_idx << ";" << std::endl;
			INDT_1 << "}" << std::endl;

			offset += input_tensor->data_dim[axis];
		}
	}

	void resolve(void) override
	{
		if (get_number_of_inputs() == 1) {
			LOG(WARNING) << "Concat node " << onnx_name << " has only one input." << std::endl;
		}

		if (axis < 0)
			axis = get_input_tensor(0)->data_dim.size() + axis;

		size_t output_axis_size = 0;
		std::vector<int> dims = get_input_tensor(0)->data_dim;

		LOG(TRACE) << "Concatenating on axis " << axis << std::endl;
		for (size_t i = 0; i < get_number_of_inputs(); i++) {
			if (get_input_tensor(0)->rank() != get_input_tensor(i)->rank()) {
				LOG(DEBUG) << "Input " << get_input_tensor(0)->name << " has " << get_input_tensor(0)->rank() << " dimensions" << std::endl;
				LOG(DEBUG) << "Input " << get_input_tensor(i)->name << " has " << get_input_tensor(i)->rank() << " dimensions" << std::endl;
				ERROR("Concat expects all inputs to have equal number of dimensions");
			}
			for (size_t j = 0; j < dims.size(); j++) {
				if (dims[j] != get_input_tensor(i)->data_dim[j] && (int)j != axis)
					ERROR("Concat's input tensors must have the same shape, except for the "
					      "dimension size of the axis to concatenate on.");
			}

			name_input(i, "input_" + std::to_string(i));
			output_axis_size += get_input_tensor(i)->data_dim[axis];
		}

		auto* rv = new Tensor;
		rv->data_dim = dims;
		rv->data_dim[axis] = output_axis_size;
		rv->data_type = get_input_tensor(0)->data_type;
		register_output(rv, "concat_result");
	}
};
} // namespace toC
