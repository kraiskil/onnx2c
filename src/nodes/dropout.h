/* This file is part of onnx2c.
 *
 * Dropout node.
 * Seems to be intended for training a network.
 * In inference this should be a pass-through node.
 * Since onnx2c is not concerned about training, this
 * is partially implemented only.
 * Technically it is possible to pass "training_mode"=true,
 * and have dropouts in inference too, but this is unimplemented.
 */
namespace toC {

class Dropout : public Node {
	public:
	Dropout() {
		op_name = "Dropout";
		seed_given=false;
	}
	/* Node attributes */
	int seed;
	bool seed_given; //not an attribute

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			if( a.name() == "seed" ) {
				seed = parse_attribute_int(a);
				seed_given=true;
			}
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		const Tensor *data = get_input_tensor(0);
		std::string datatype = data->data_type_str();
		dst << "\t/* Dropout */" << std::endl;

		dst << "\t" << datatype << " *in = (" << datatype << "*)input;" << std::endl;
		dst << "\t" << datatype << " *out = (" << datatype << "*)output;" << std::endl;
		if( is_output_N_used(1) )
			dst << "\t" << "bool *mask_1d = (bool*)mask;" << std::endl;

		dst << "\tfor( uint32_t d=0; d<" << data->data_num_elem() << "; d++) {" << std::endl;

		dst << "\t\t" << "out[d] = in[d];" << std::endl;

		// NB: specifications say mask can have 'false' values only when training the network.
		if( is_output_N_used(1) )
			dst << "\t\t" << "mask_1d[d] = true;" << std::endl;
		dst << "\t}" << std::endl;
	}


	virtual void resolve(void) override
	{
		const Tensor *data = get_input_tensor(0);
		name_input(0, "input");
		if(  typeConstraint_highPrecisionNumeric(data) == false )
			ERROR("Incorrect input for node");

		if( get_number_of_inputs() > 1 ) {
			name_input(1, "ratio");
		}

		if( get_number_of_inputs() > 2 ) {
			ERROR("Unimplemented - training_mode input to Dropout");
			name_input(2, "training_mode");
		}

		/* Create output tensor */
		Tensor *rv = new Tensor;
		rv->data_dim = data->data_dim;
		rv->data_type = data->data_type;
		register_output(rv, "output");

		/* Mask is optional  */
		if( is_output_N_used(1) )
		{
			rv = new Tensor;
			rv->data_dim = data->data_dim;
			rv->data_type = onnx::TensorProto_DataType_BOOL;
			register_output(rv, "mask");
		}
	}
};
}
