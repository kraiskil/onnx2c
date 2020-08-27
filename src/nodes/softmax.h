/* This file is part of onnx2c.
 * 
 * Softmax.
 * Calculates
 * exp( x - max ) / sum( exp( x - max ) )
 * sum() is calculated over batches consisting of all
 * dimensions 'deeper than'/'more inner than'/'right of'
 * given attribute 'axis'.
 */ 
namespace toC {

class Softmax : public Node {
	public:
	Softmax() {
		op_name = "Softmax";
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

	virtual void print(std::ostream &dst) const
	{
		const Tensor *input = inputs[0];
		const Tensor *output = outputs[0];
		std::string type = input->data_type_str();
		unsigned n_dim = input->data_dim.size();
		std::string expfunc = "expf";
		if( type == "double" )
			expfunc = "exp";
		//TODO fp16?

		unsigned flatten_axis;
		if( axis < 0 ) 
			flatten_axis = input->data_dim.size() + axis;
		else
			flatten_axis = axis;

		dst << "\t/* Softmax" << std::endl;
		dst << "\t * axis = " << axis << std::endl;
		dst << "\t */" << std::endl; 
		dst << "\t" << type << " *getmax = (" << type << "*)" << input->cname() << ";" << std::endl;
		dst << "\t" << type << " sum = 0.0;" << std::endl;
		dst << "\t" << type << " max = -INFINITY;" << std::endl;
		dst << std::endl;

		std::string idxs;
		for( unsigned i = 0; i<n_dim; i++)
			idxs += "[i" + std::to_string(i) + "]";

		for( unsigned i = 0; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// Calculate max of the flattened inner dimensions, and close those loops
		dst << "\t\t" << "max = max>" << input->cname() << idxs << " ? max :" << input->cname() << idxs << ";" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		// Calculate exp() and running sum of innermost flattened dimensions
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		dst << "\t\t" << output->cname() << idxs << " = ";
		dst           << expfunc << "(" << input->cname() << idxs << "-max);" << std::endl;
		dst << "\t\t" << "sum += " << output->cname() << idxs << ";" << std::endl; 
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;
			
		// loop again over inner dimensions to do the division
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		dst << "\t\t" << output->cname() <<idxs<<" /= sum;" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		dst << "\t" << "sum = 0.0;" << std::endl; 
		dst << "\t" << "max = -INFINITY;" << std::endl; 

		
		for( unsigned i = 0; i<flatten_axis; i++ )
			dst << "\t}" << std::endl;
	}



	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Softmax");

		const Tensor *A = inputs[0];
		if( typeConstraint_allFloatingPoints(A) == false)
			ERROR("Incorrect input for node");

		Tensor *rv = new Tensor;
		rv->data_dim = A->data_dim;
		rv->data_type = A->data_type;
		outputs.push_back(rv);
	}
};
}
