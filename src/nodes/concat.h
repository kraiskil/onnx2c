
namespace toC {

    class Concat : public Node {
    public:
        Concat() {
            op_name = "Concat";
            output = nullptr;
        }

        int axis = 1;
        std::vector<const Tensor *> node_inputs;
        const Tensor *output;

        void print_parameters(std::ostream &dst, bool decorate) const override {
            size_t input_count = node_inputs.size();
            for (size_t i = 0; i < input_count; i++) {
                node_inputs[i]->print_tensor(dst, !decorate);
                if (i < (input_count - 1))
                    dst << ", ";
            }
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

            int64_t input_count = node_inputs.size();
            int64_t axis_value = axis;

            dst << "size_t input_count = " << input_count << ";";
            dst << "size_t axis_value = " << axis_value << ";";

            // the axisPitch is the number of elements to add to move
            // to the next split axis in the output
            int64_t axisPitch = 1;
            for (int i = output->data_dim.size() - 1; i >= axis; i--) {
                axisPitch *= output->data_dim[i];
            }

            int64_t outputBase = 0;
            dst << "int64_t outputOffset = 0;" << std::endl;
            for (int64_t inputIndex = 0; inputIndex < input_count; inputIndex++) {

                // the inputAxisPitch is the number of elements to add
                // to move to the next split axis in the input

                int64_t inputAxisPitch = 1;
                for (int i = node_inputs[inputIndex]->data_dim.size() - 1; i >= axis; i--) {
                    inputAxisPitch *= node_inputs[inputIndex]->data_dim[i];
                }

                int64_t inputSize = node_inputs[inputIndex]->data_num_elem();

                // copy the data across.
                // for every 'inputAxisPitch' values copied, we move over by
                // the 'axisPitch'

                dst << "outputOffset = " << outputBase << ";" << std::endl;
                dst << "for (int64_t i = 0, j = 0; i < " << inputSize << "; i++) {" << std::endl;
                dst << "    *((" << output->data_type_str() << "*)" << output->cname() << " + (outputOffset + i)) = ";
                dst << "    *((" << output->data_type_str() << "*)" << node_inputs[inputIndex]->cname() << " + i);";
                dst << std::endl;
                dst << "    if (++j == " << inputAxisPitch << ") {" << std::endl;
                dst << "        outputOffset += (" << axisPitch - inputAxisPitch << ");" << std::endl;
                dst << "        j = 0;" << std::endl;
                dst << "    }" << std::endl;
                dst << "}" << std::endl;

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
            output = rv;
            outputs.push_back(rv);
        }
    };
}
