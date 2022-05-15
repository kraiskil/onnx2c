/* This file is part of onnx2c.
 *
 * Transpose and generic permutation of a tensor. 
 */ 
namespace toC {

class Transpose : public Node {
	public:
	Transpose() {
		op_name = "Transpose";
		data=transposed=NULL;
	}
	std::vector<int> perm;

	// inputs
	const Tensor *data;
	// outputs
	const Tensor *transposed;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		transposed->print_tensor(dst, !decorate);
	}


	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
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

	virtual void print(std::ostream &dst) const override
	{
		std::string type = data->data_type_str();
		unsigned n_dim = data->data_dim.size();


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
			dst <<               idx << "<" << data->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// copy data
		dst << "\t\t" << transposed->cname() << out_idx << " = ";
		dst <<           data->cname() << in_idx << ";" << std::endl;

		// close loops
		for( unsigned i = 0; i<n_dim; i++)
			dst << "\t}" << std::endl;
	}



	virtual void resolve(void) override
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Transpose");

		data = inputs[0];
		unsigned n_dim = data->data_dim.size();

		// "By default, reverse the dimensions, otherwise permute the axes according to the values given."
		std::vector<int> out_dim;
		if( perm.size() == 0 )
			for( unsigned i=0; i<n_dim; i++)
				perm.push_back( n_dim - 1 - i );
		for( int d : perm )
			out_dim.push_back( data->data_dim[d] );

		Tensor *rv = new Tensor;
		rv->data_dim = out_dim;
		rv->data_type = data->data_type;
		transposed = rv;
		outputs.push_back(rv);
	}
};
}

