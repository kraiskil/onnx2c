/* This file is part of onnx2c.
 *
 * Gather node.
 */
namespace toC {

class Gather : public Node {
	public:
	Gather() {
		op_name = "Gather";
		axis=0;
	}
	/* Node attributes */
	int axis;

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "axis" )
				axis = parse_attribute_int(a);
			else
				ERROR("Unknown attribute for Gather: " + a.name());
		}
	}


	virtual void resolve(void) override
	{
		const Tensor *data = get_input_tensor(0);
		const Tensor *indices = get_input_tensor(1);
		name_input(0, "X");
		name_input(1, "indices");

		unsigned a = axis >= 0 ? axis : data->rank()+axis;
		assert(a < data->rank());

		Tensor *t = new Tensor;

		// output shape = data.shape[:axis] + indices.shape + data.shape[axis+1:]
		for (unsigned i = 0; i < a; i++)
			t->data_dim.push_back(data->data_dim[i]);
		for (unsigned i = 0; i < indices->rank(); i++)
			t->data_dim.push_back(indices->data_dim[i]);
		for (unsigned i = a+1; i < data->rank(); i++)
			t->data_dim.push_back(data->data_dim[i]);

		t->data_type = data->data_type;
		register_output(t, "Y");
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *data = get_input_tensor(0);
		const Tensor *indices = get_input_tensor(1);
		const Tensor *output= get_output_tensor(0);
		INDT_1 << "/* Gather" << std::endl;
		INDT_1 << "   axis = " << axis << std::endl;
		INDT_1 << " */" << std::endl;

		// The real axis number, counting from 0
		unsigned a = axis >= 0 ? axis : data->rank()+axis;
		
		std::string oidx = output->rank() == 0 ? "*Y" : "Y";
		for (unsigned r = 0; r < output->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (size_t " << lv << "=0; ";
			dst << lv << "<" << output->data_dim[r] << "; ";
			dst << lv << "++)" << std::endl;

			oidx += "[" + lv + "]";
		}

		std::string didx = "X";
		for (unsigned r = 0; r < a; r++) {
			didx += "[i" + std::to_string(r) + "]";
		}
		didx += "[idx]";
		for (unsigned r = a + indices->rank(); r < output->rank(); r++) {
			didx += "[i" + std::to_string(r) + "]";
		}

		std::string iidx = indices->rank() == 0 ? "*indices" : "indices";
		for (unsigned r = 0; r < indices->rank(); r++) {
			iidx += "[i" + std::to_string(r + a) + "]";
		}

		INDT_1 << "{" << std::endl;
		INDT_2 << "int32_t idx = " << iidx << ";" << std::endl;
		INDT_2 << "idx = idx < 0 ? " << data->data_dim[a] << "+idx : idx;" << std::endl;
		INDT_2 << oidx << " = " << didx << ";" << std::endl;
		INDT_1 << "}" << std::endl;
	}
};
}

