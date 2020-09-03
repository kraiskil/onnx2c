
namespace toC {

class Add : public Node {
	public:
	Add() {
		op_name = "Add";
		A=B=C=NULL;
	}
	const Tensor *A;
	const Tensor *B;
	const Tensor *C;


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		A->print_tensor(dst, !decorate);
		dst << ", ";
		B->print_tensor(dst, !decorate);
		dst << ", ";
		C->print_tensor(dst, !decorate);
	}

	virtual void print(std::ostream &dst) const override
	{
		std::string type = A->data_type_str();

		dst << "\t/* Add*/" << std::endl;

		/* Simple case where no broadcasting is needed */
		/* TODO: This check is not sufficient: [1][1] and [1][8] needs a broadcast */
		if( A->data_num_elem() == B->data_num_elem()) {
			dst << "\t" << type << " *A = (" << type << "*)" << A->cname() << ";" << std::endl;
			dst << "\t" << type << " *B = (" << type << "*)" << B->cname() << ";" << std::endl;
			dst << "\t" << type << " *C = (" << type << "*)" << C->cname() << ";" << std::endl;

			dst << "\t" << "for( uint32_t i=0; i<" << A->data_num_elem() << "; i++ )" << std::endl;
			dst << "\t\tC[i] = A[i] + B[i];" << std::endl;
			dst << std::endl;
		}
		else {
		dst << "\t/* multidimensional broadcast */" << std::endl;
			std::vector<int> result_dim;
			multidirectional_broadcast_size(A->data_dim, B->data_dim, result_dim);
			uint32_t apad = result_dim.size() - A->data_dim.size();
			uint32_t bpad = result_dim.size() - B->data_dim.size();
			std::string indent="\t";
			// save the indices into the tensors. Must do this, when
			// adding tensors of different dimensionality.
			std::vector<std::string> cloops;
			std::vector<std::string> aloops;
			std::vector<std::string> bloops;
			for( uint32_t dim=0; dim<result_dim.size(); dim++) {
				std::string loop = "i" + std::to_string(dim);
				cloops.push_back(loop);
				dst << indent << "for( uint32_t " << loop<<"=0; ";
				dst <<             loop << "<" << result_dim[dim] <<"; ";
				dst <<             loop<<"++ ) {" << std::endl;
				indent +="\t";

				if( dim >= apad ) {
					std::string aloop = (A->data_dim[dim-apad] == 1 ? "0" : loop);
					aloops.push_back( aloop );
				}
				else
					aloops.push_back("-");
				if( dim >= bpad ) {
					std::string bloop = (B->data_dim[dim-bpad] == 1 ? "0" : loop);
					bloops.push_back( bloop );
				}
				else
					bloops.push_back("-");
			}

			/* Print the line with addition */
			dst << indent << C->cname();
			for( std::string cloop : cloops )
				dst << "[" << cloop << "]";
			dst << " = " << A->cname();
			for( std::string aloop : aloops )
				if( aloop != "-" )
					dst << "[" << aloop << "]";
			dst << " + " << B->cname();
			for( std::string bloop : bloops )
				if( bloop != "-" )
					dst << "[" << bloop << "]";
			dst << ";" << std::endl;

			for( uint32_t dim=0; dim<result_dim.size(); dim++) {
				dst << "\t}" << std::endl;
			}
		}
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() != 2 )
			ERROR("wrong number of inputs to Add");

		A = inputs[0];
		B = inputs[1];
		if(  typeConstraint_highPrecisionNumeric(A) == false
		   ||typeConstraint_highPrecisionNumeric(B) == false)
			ERROR("Incorrect input for node"); 


		std::vector<int> result_dim;
		multidirectional_broadcast_size(A->data_dim, B->data_dim, result_dim);

		uint64_t total_elems=1;
		for( auto d : result_dim )
			total_elems *= d;

		Tensor *rv = new Tensor;
		rv->data_dim = result_dim;
		rv->data_type = A->data_type;
		C=rv;
		outputs.push_back(rv);
	}
};
}
