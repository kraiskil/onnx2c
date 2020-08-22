/* This file is part of onnx2c.
 *
 * Transpose and generic permutation of a tensor. 
 */ 
namespace toC {

class Transpose : public Node {
	public:
	Transpose() {
		op_name = "Transpose";
	}
	std::vector<int> perm;

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto a : node.attribute() ) {
			if( a.name() == "perm" ) {
				if( a.type() != onnx::AttributeProto_AttributeType_INTS )
					ERROR("Bad attribute " << a.name());

				for( auto i : a.ints() ) {
					perm.push_back(i);
				}
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


		dst << "\t/* Transpose" << std::endl;
		dst << "\t * perm = ";
		for( int p : perm )
			dst << std::to_string(p) << " ";
		dst << std::endl; 
		dst << "\t */" << std::endl; 

		std::string in_idx, out_idx;
		for( unsigned i = 0; i<n_dim; i++) {
			in_idx += "[i" + std::to_string(i) + "]";
			out_idx += "[i" + std::to_string(perm[i]) + "]";
		}

		// loop over all indices
		for( unsigned i = 0; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// copy data
		dst << "\t\t" << output->cname() << out_idx << " = ";
		dst <<           input->cname() << in_idx << ";" << std::endl;

		// close loops
		for( unsigned i = 0; i<n_dim; i++)
			dst << "\t}" << std::endl;
	}



	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Transpose");

		const Tensor *A = inputs[0];
		unsigned n_dim = A->data_dim.size();

		// "By default, reverse the dimensions, otherwise permute the axes according to the values given."
		std::vector<int> out_dim;
		if( perm.size() == 0 )
			for( unsigned i=0; i<n_dim; i++)
				perm.push_back( n_dim - 1 - i );
		for( int d : perm )
			out_dim.push_back( A->data_dim[d] );

		Tensor *rv = new Tensor;
		rv->data_dim = out_dim;
		rv->data_type = A->data_type;
		rv->data_num_elem = A->data_num_elem;
		outputs.push_back(rv);
	}
};
}

