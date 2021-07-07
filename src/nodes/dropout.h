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
		/* TODO: initialize class variables (attributes and tensors) */
		data=ratio=training_mode=output=mask=NULL;
		seed_given=false;
	}
	/* Node attributes */
	int seed;
	bool seed_given; //not an attribute

	/* input */
	const Tensor *data;
	/* optional inputs */
	const Tensor *ratio;
	const Tensor *training_mode;
	/* output */
	const Tensor *output;
	/* optional output */
	const Tensor *mask;

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			if( a.name() == "seed" ) {
				seed = parse_attribute_int(a);
				seed_given=true;
			}
		}
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";

		if( ratio ) {
			ratio->print_tensor_as_const(dst, !decorate);
			dst << ", ";
		}
		if( training_mode ) {
			training_mode->print_tensor_as_const(dst, !decorate);
			dst << ", ";
		}


		output->print_tensor(dst, !decorate);

		if( mask ) {
			dst << ", ";
			mask->print_tensor(dst, !decorate);
		}
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		std::string datatype = data->data_type_str();
		dst << "\t/* Dropout */" << std::endl;

		dst << "\t" << datatype << " *in = (" << datatype << "*)" << data->cname() << ";" << std::endl;
		dst << "\t" << datatype << " *out = (" << datatype << "*)" << output->cname() << ";" << std::endl;
		if( mask )
			dst << "\t" << "bool *mask = (bool*)" << mask->cname() << ";" << std::endl;

		dst << "\tfor( uint32_t d=0; d<" << data->data_num_elem() << "; d++) {" << std::endl;

		dst << "\t\t" << "out[d] = in[d];" << std::endl;

		// NB: specifications say mask can have 'false' values only when training the network.
		if( mask )
			dst << "\t\t" << "mask[d] = true;" << std::endl;
		dst << "\t}" << std::endl;
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		data = inputs[0];
		if(  typeConstraint_highPrecisionNumeric(data) == false )
			ERROR("Incorrect input for node");

		if( inputs.size() > 1 )
			ratio = inputs[1];

		if( inputs.size() > 2 ) {
			ERROR("Unimplemented - training_mode input to Dropout");
			training_mode = inputs[2];
		}

		/* Create output tensor */
		Tensor *rv = new Tensor;
		rv->data_dim = data->data_dim;
		rv->data_type = data->data_type;
		output=rv;
		outputs.push_back(rv);

		/* Mask is optional  */
		if( is_output_N_used(1) )
		{
			rv = new Tensor;
			rv->data_dim = data->data_dim;
			rv->data_type = onnx::TensorProto_DataType_BOOL;
			mask=rv;
			outputs.push_back(rv);
		}
	}
};
}
