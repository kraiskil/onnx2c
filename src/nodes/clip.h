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
		input=min_tensor=max_tensor=output=NULL;
		min_attr = std::numeric_limits<float>::lowest();
		max_attr = std::numeric_limits<float>::max();
	}

	// Early versions (opset v.6) used attributes to pass the clipping limits.
	// An implementation where these values are used if the tensors are not given
	// works for all versions of Clip
	float min_attr, max_attr;

	const Tensor *input;
	const Tensor *min_tensor;
	const Tensor *max_tensor;
	const Tensor *output;


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


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		input = inputs[0];

		if (inputs.size() > 1 && inputs[1]->is_used())
			min_tensor = inputs[1];
		if (inputs.size() > 2 && inputs[2]->is_used())
			max_tensor = inputs[2];


		Tensor *t = new Tensor;
		t->data_dim = input->data_dim;
		t->data_type = input->data_type;
		/* Store the created tensor both as reference in this node, and into
		 * the return value vector! */
		output = t;
		outputs.push_back(t);
	}


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		input->print_tensor_as_const(dst, !decorate);

		if (min_tensor) {
			dst << ", ";
			min_tensor->print_tensor_as_const(dst, !decorate);
		}

		if (max_tensor) {
			dst << ", ";
			max_tensor->print_tensor_as_const(dst, !decorate);
		}

		dst << ", ";
		output->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Clip */" << std::endl;

		if( min_tensor )
			INDT_1 << min_tensor->data_type_str() << " minv = " << min_tensor->cname() << "[0];" << std::endl;
		else
			INDT_1 << "float minv = " << min_attr << ";" << std::endl;

		if( max_tensor )
			INDT_1 << max_tensor->data_type_str() << " maxv = " << max_tensor->cname() << "[0];" << std::endl;
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

		INDT_2 << output->cname() << idx << " = ";
		 dst << "MAX( MIN( " << input->cname() << idx << ", maxv), minv);" << std::endl;

		for( unsigned r=0; r<input->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}

	}
};
}

