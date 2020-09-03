/* This file is part of onnx2c.
 * 
 * Sigmoid.
 * Calculates element vise:
 * y = 1 / (1 + exp(-x))
 */
namespace toC {

class Sigmoid : public Node {
	public:
	Sigmoid() {
		op_name = "Sigmoid";
		X=Y=NULL;
	}

	// inputs
	const Tensor *X;
	// outputs
	const Tensor *Y;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor(dst, !decorate);
		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{
		std::string type = X->data_type_str();
		std::string expfunction;
		std::string castlabel;
		if( type == "double" ) {
			castlabel="";
			expfunction="exp";
		}
		else { // assume float - how about fp16?
			castlabel="f";
			expfunction="expf";
		}

		dst << "\t/* Sigmoid*/" << std::endl;

		dst << "\t" << type << " *input = (" << type << "*)" << X->cname() << ";" << std::endl;
		dst << "\t" << type << " *output = (" << type << "*)" << Y->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << X->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\toutput[i] = 1.0"<<castlabel<< "/(1+" << expfunction << "(-input[i]));" << std::endl;
		dst << std::endl;
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Sigmoid");

		X = inputs[0];

		Tensor *rv = new Tensor;
		rv->data_dim = X->data_dim;
		rv->data_type = X->data_type;
		Y = rv;
		outputs.push_back(rv);
	}
};
}

