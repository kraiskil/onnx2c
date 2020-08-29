/* This file is part of onnx2c.
 * 
 * BatchNormalization.
 * Algorithm as "described in this paper https://arxiv.org/abs/1502.03167"
 *
 * i.e. This operator provides a "whitening" of the data in the middle of
 * the network, (as opposed to just a preprocessing the input).
 * 
 * Algorithm calculated is: 
 *
 * 	y = scale * X + bias
 * where
 * 	X = (x - mean) / sqrt( variance + epsilon )
 *
 *
 * mean and variance can, optionally, be updated and sent as output,
 * but this is still unimplemented in onnx2c.
 */ 
namespace toC {

class BatchNormalization : public Node {
	public:
	BatchNormalization() {
		op_name = "BatchNormalization";
		epsilon = 1e-5;
		momentum = 0.9;
	}
	float epsilon;
	float momentum;


	virtual void parseAttribute_epsilon( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_FLOAT )
			ERROR("Bad attribute " << a.name());
		if( a.has_f() == false )
			ERROR("Bad attribute " << a.name());
		epsilon = a.f();
	}
	virtual void parseAttribute_momentum( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_FLOAT )
			ERROR("Bad attribute " << a.name());
		if( a.has_f() == false )
			ERROR("Bad attribute " << a.name());
		momentum = a.f();
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto a : node.attribute() ) {
			if( a.name() == "epsilon" )
				parseAttribute_epsilon(a);
			else if( a.name() == "momentum" )
				parseAttribute_momentum(a);
			else
				ERROR("Unknown attribute " << a.name());
		}
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *input = inputs[0];
		const Tensor *scale = inputs[1];
		const Tensor *bias = inputs[2]; // "B" in spec
		const Tensor *mean = inputs[3];
		const Tensor *var = inputs[4];
		const Tensor *output = outputs[0];

		int batch_size =input->data_dim[0]; 
		int num_chan =input->data_dim[1]; 
		std::string type = input->data_type_str();


		dst << "\t/* BatchNormalization" << std::endl;
		dst << "\t * epsilon = " << epsilon << std::endl;
		dst << "\t * momentum = " << momentum << std::endl;
		dst << "\t */" << std::endl << std::endl;

		dst<<"\t" << "for( int32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		dst<<"\t" << "for( int32_t c=0; c<" << num_chan << "; c++ ) {" << std::endl;

		// create the indexing string for picking out an element in input/output
		std::string idxs = "[b][c]";
		for( unsigned i = 2; i<input->data_dim.size(); i++)
			idxs += "[i" + std::to_string(i) + "]";


		// Loop over data dimensions
		for( unsigned i = 2; i<input->data_dim.size(); i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		dst << "\t\t" << type << " tmp_X = ";
		dst <<           "( " << input->cname() << idxs << " - " << mean->cname() << "[c] ) / ";
		dst <<           "( sqrt(" << var->cname() << "[c] + " << epsilon << ") );" << std::endl;

		dst << "\t\t" << output->cname() << idxs << " = ";
		dst <<           "tmp_X * " << scale->cname() << "[c] + " << bias->cname() << "[c];" << std::endl;

		for( unsigned i = 2; i<input->data_dim.size(); i++)
			dst << "\t}" << std::endl;

		dst<<"\t" << "}" << std::endl;
		dst<<"\t" << "}" << std::endl;
	}



	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() != 5 )
			ERROR("wrong number of inputs to BatchNormalization");

		const Tensor *X = inputs[0];
		const Tensor *scale = inputs[1];
		const Tensor *bias = inputs[2]; // "B" in spec
		const Tensor *mean = inputs[3];
		const Tensor *var = inputs[4];

		if( typeConstraint_plainFloatingPoints(X) == false)
			ERROR("Incorrect input for node");
		if( typeConstraint_plainFloatingPoints(scale) == false)
			ERROR("Incorrect input for node");
		if( typeConstraint_plainFloatingPoints(bias) == false)
			ERROR("Incorrect input for node");
		if( typeConstraint_plainFloatingPoints(mean) == false)
			ERROR("Incorrect input for node");
		if( typeConstraint_plainFloatingPoints(var) == false)
			ERROR("Incorrect input for node");

		Tensor *rv = new Tensor;
		rv->data_dim = X->data_dim;
		rv->data_type = X->data_type;
		outputs.push_back(rv);
	}
};
}

