/* This file is part of onnx2c.
 *
 * Clip node.
 * Limits input to be between provided boundaries
 */
namespace toC {

class Clip : public Node {
	public:
	Clip() {
		op_name = "Clip";
		min_attr = std::numeric_limits<float>::lowest();
		max_attr = std::numeric_limits<float>::max();
	}

	// Early versions (opset v.6) used attributes to pass the clipping limits.
	// An implementation where these values are used if the tensors are not given
	// works for all versions of Clip
	float min_attr, max_attr;


	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "min" )
				min_attr = parse_attribute_float(a);
			else if( a.name() == "max" )
				max_attr = parse_attribute_float(a);
			else
				ERROR("Unknown attribute for Clip: "+a.name());
		}
	}


	virtual void resolve(void) override
	{
		const Tensor *input = inputs[0];
		register_input(inputs[0], "input");
		if (inputs.size() > 1 && inputs[1]->is_used())
			register_input(inputs[1], "min_tensor");
		if (inputs.size() > 2 && inputs[2]->is_used())
			register_input(inputs[2], "max_tensor");

		Tensor *t = new Tensor;
		t->data_dim = input->data_dim;
		t->data_type = input->data_type;
		register_output(t, "output");
	}

	virtual void print(std::ostream &dst) const override
	{

		const Tensor *input = inputs[0];
		const Tensor *min_tensor = nullptr;
		const Tensor *max_tensor = nullptr;

		if (inputs.size() > 1 && inputs[1]->is_used())
			min_tensor = inputs[1];
		if (inputs.size() > 2 && inputs[2]->is_used())
			max_tensor = inputs[2];

		INDT_1 << "/* Clip */" << std::endl;

		if( min_tensor )
			INDT_1 << min_tensor->data_type_str() << " minv = min_tensor[0];" << std::endl;
		else
			INDT_1 << "float minv = " << min_attr << ";" << std::endl;

		if( max_tensor )
			INDT_1 << max_tensor->data_type_str() << " maxv = max_tensor[0];" << std::endl;
		else
			INDT_1 << "float maxv = " << max_attr << ";" << std::endl;

		std::string idx = "";
		for( unsigned r=0; r< input->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
			 dst << lv << "<" << input->data_dim[r] << "; ";
			 dst << lv << "++) {" << std::endl;

			idx += "[" + lv + "]";
		}

		INDT_2 << "output" << idx << " = ";
		 dst << "MAX( MIN( input"<< idx << ", maxv), minv);" << std::endl;

		for( unsigned r=0; r<input->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
}

