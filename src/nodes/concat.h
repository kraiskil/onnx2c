
namespace toC {

	class Concat : public Node {
	public:
		Concat() {
			op_name = "Concat";
			axis = 1;
			concat_result  = nullptr;
		}

		// inputs
		std::vector<const Tensor *> node_inputs; // 'inputs' in the spec

		// output
		const Tensor *concat_result ;

		// attribute
		int axis;

		void print_parameters(std::ostream &dst, bool decorate) const override {
			size_t input_count = node_inputs.size();
			for (size_t i = 0; i < input_count; i++) {
				node_inputs[i]->print_tensor_as_const(dst, !decorate);
				dst << ", ";
			}
			concat_result->print_tensor(dst, !decorate);
		}

		void parseAttributes(onnx::NodeProto &node) override {
			for (const auto &a : node.attribute()) {
				if (a.name() == "axis") {
					if (a.type() != onnx::AttributeProto_AttributeType_INT)
						ERROR("Bad attribute " << a.name());
					axis = a.i();
				} else
					ERROR("Unknown attribute " << a.name());
			}
		}

		void print(std::ostream &dst) const override {

			dst << "\t/* Concat */" << std::endl;

			// the axisPitch is the number of elements to add to move to the next split axis in the concat_result
			int64_t axisPitch = 1;
			for (int i = concat_result ->data_dim.size() - 1; i >= axis; i--) {
				axisPitch *= concat_result ->data_dim[i];
			}

			dst << "\tint64_t outputOffset;" << std::endl;

			int64_t outputBase = 0;
			int64_t input_count = node_inputs.size();

			for (int64_t inputIndex = 0; inputIndex < input_count; inputIndex++) {

				// the inputAxisPitch is the number of elements to add to move to the next split axis in the inputs
				int64_t inputAxisPitch = 1;
				for (int i = node_inputs[inputIndex]->data_dim.size() - 1; i >= axis; i--) {
					inputAxisPitch *= node_inputs[inputIndex]->data_dim[i];
				}

				int64_t inputSize = node_inputs[inputIndex]->data_num_elem();

				// copy the data across: for every 'inputAxisPitch' values copied, we move over by the 'axisPitch'
				dst << "\toutputOffset = " << outputBase << ";" << std::endl;
				dst << "\tfor (int64_t i = 0, j = 0; i < " << inputSize << "; i++) {" << std::endl;

				dst << "\t\t*((" << concat_result ->data_type_str() << "*)" << concat_result ->cname();
				dst << " + (outputOffset + i)) = *((" << concat_result ->data_type_str() << "*)";
				dst << node_inputs[inputIndex]->cname() << " + i);" << std::endl;

				dst << "\t\tif (++j == " << inputAxisPitch << ") {" << std::endl;
				dst << "\t\t\toutputOffset += (" << axisPitch - inputAxisPitch << ");" << std::endl;
				dst << "\t\t\tj = 0;" << std::endl;
				dst << "\t\t}" << std::endl;

				dst << "\t}" << std::endl;

				outputBase += inputAxisPitch;
			}

		}

		void resolveOutput(const std::vector<const Tensor *> &inputs, std::vector<Tensor *> &outputs) override {
			node_inputs = inputs;
			if (inputs.size() < 2)
				ERROR("Wrong number of inputs to Concat");

			if (axis < 0)
				axis = inputs[0]->data_dim.size() + axis;

			auto *rv = new Tensor;
			rv->data_dim = inputs[0]->data_dim;
			size_t input_count = node_inputs.size();
			size_t output_axis_size = 0;
			size_t i, j;
			std::vector<int> dims = inputs[0]->data_dim;
			for (i = 0; i < input_count; i++) {
				for (j = 1; j < dims.size(); j++) {
					if (dims[j] != inputs[i]->data_dim[j] && (int) j != axis)
						ERROR("Concat's input tensors must have the same shape, except for the "
							  "dimension size of the axis to concatenate on.");
				}
				output_axis_size += inputs[i]->data_dim[axis];
			}
			rv->data_dim[axis] = output_axis_size;
			rv->data_type = inputs[0]->data_type;
			concat_result  = rv;
			outputs.push_back(rv);
		}
	};
}
