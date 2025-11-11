
namespace toC {

class Flatten : public Node {
	public:
	Flatten() {
		op_name = "Flatten";
		axis = 1;
	}
	int axis;

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
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
		const Tensor *input = get_input_tensor(0);
		std::string type = input->data_type_str();

		dst << "\t/* Flatten*/" << std::endl;

		dst << "\t" << type << " *input_ = (" << type << "*)input;" << std::endl;
		dst << "\t" << type << " *output_ = (" << type << "*)output;" << std::endl;

		dst << "\t" << "for( size_t i=0; i<" << input->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\toutput_[i] = input_[i];" << std::endl;
		dst << std::endl;
	}


	virtual void resolve(void) override
	{
		if( get_number_of_inputs() != 1 )
			ERROR("wrong number of inputs to Flatten");

		const Tensor *input = get_input_tensor(0);
		name_input(0, "input");

		// output:
		// A 2D tensor with the contents of the input tensor, with input dimensions up to axis
		// flattened to the outer dimension of the output and remaining input dimensions flattened
		// into the inner dimension of the output.
		//
		// "flatten" means merging here
		std::vector<int> result_dim;

		int count_axis = axis;
		if( axis < 0 ) 
			count_axis = input->data_dim.size() + axis;

		int dim=1;
		int i;
		for(i=0; i<count_axis; i++)
			dim *= input->data_dim[i];
		result_dim.push_back(dim);
		dim = 1;
		for(; i<(int)input->data_dim.size(); i++)
			dim *= input->data_dim[i];
		result_dim.push_back(dim);

		Tensor *rv = new Tensor;
		rv->data_dim = result_dim;
		rv->data_type = input->data_type;
		register_output(rv, "output");
	}
};
}
