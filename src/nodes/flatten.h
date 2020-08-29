
namespace toC {

class Flatten : public Node {
	public:
	Flatten() {
		op_name = "Flatten";
		axis = 1;
	}
	int axis;

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto a : node.attribute() ) {
			if( a.name() == "axis" ) {
				if( a.type() != onnx::AttributeProto_AttributeType_INT )
					ERROR("Bad attribute " << a.name());
				if( a.has_i() == false )
					ERROR("Bad attribute " << a.name());
				axis = a.i();
			}
			else
				ERROR("Unknown attribute " << a.name());
		}
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *input = inputs[0];
		const Tensor *output = outputs[0];
		std::string type = input->data_type_str();

		dst << "\t/* Flatten*/" << std::endl;

		dst << "\t" << type << " *input = (" << type << "*)" << input->cname() << ";" << std::endl;
		dst << "\t" << type << " *output = (" << type << "*)" << output->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << input->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\toutput[i] = input[i];" << std::endl;
		dst << std::endl;
	}



	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Flatten");

		const Tensor *A = inputs[0];

		// output:
		// A 2D tensor with the contents of the input tensor, with input dimensions up to axis
		// flattened to the outer dimension of the output and remaining input dimensions flattened
		// into the inner dimension of the output.
		//
		// "flatten" means merging here
		std::vector<int> result_dim;

		int count_axis = axis;
		if( axis < 0 ) 
			count_axis = A->data_dim.size() + axis;

		int dim=1;
		int i;
		for(i=0; i<count_axis; i++)
			dim *= A->data_dim[i];
		result_dim.push_back(dim);
		dim = 1;
		for(; i<(int)A->data_dim.size(); i++)
			dim *= A->data_dim[i];
		result_dim.push_back(dim);

		Tensor *rv = new Tensor;
		rv->data_dim = result_dim;
		rv->data_type = A->data_type;
		outputs.push_back(rv);
	}
};
}
