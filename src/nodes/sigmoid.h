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
	}


	virtual void print(std::ostream &dst) const
	{
		const Tensor *input = inputs[0];
		const Tensor *output = outputs[0];
		std::string type = input->data_type_str();
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

		dst << "\t" << type << " *input = (" << type << "*)" << input->cname() << ";" << std::endl;
		dst << "\t" << type << " *output = (" << type << "*)" << output->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << input->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\toutput[i] = 1.0"<<castlabel<< "/(1+" << expfunction << "(-input[i]));" << std::endl;
		dst << std::endl;
	}



	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Sigmoid");

		const Tensor *A = inputs[0];

		Tensor *rv = new Tensor;
		rv->data_dim = A->data_dim;
		rv->data_type = A->data_type;
		outputs.push_back(rv);
	}
};
}

