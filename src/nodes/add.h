
namespace toC {

class Add : public Op {
	public:
	Add() {
		name = "Add";
	}

	virtual void print(std::ostream &dst, const Node *node) const
	{
		if( node->inputs.size() != 2 )
			ERROR("wrong number of inputs to Add");
		if( node->outputs.size() != 1 )
			ERROR("wrong number of outputs from Add");
		std::string type = node->inputs[0]->data_type_str();

		dst << "\t/*Add*/" << std::endl;
		
		dst << "\t" << type << " *A = (" << type << "*)" << node->inputs[0]->cname() << ";" << std::endl;
		dst << "\t" << type << " *B = (" << type << "*)" << node->inputs[1]->cname() << ";" << std::endl;
		dst << "\t" << type << " *C = (" << type << "*)" << node->outputs[0]->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << node->inputs[0]->data_num_elem << "; i++ )" << std::endl;
		dst << "\t\tC[i] = A[i] + B[i];" << std::endl;
		dst << std::endl;
	}
 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) const
	{
		const Tensor *A = inputs[0];
		const Tensor *B = inputs[1];
		int dimA = A->data_dim[0];
		int dimB = B->data_dim[0];
		if(  typeConstraint_highPrecisionNumeric(A) == false
		   ||typeConstraint_highPrecisionNumeric(B) == false)
			ERROR("Incorrect input for node"); 

		// TODO: mess. Add allows to add [N] + [N], [1xN]+[N] and who
		// knows what other permutations... For now, [1xN] and [N] are stored
		// with identical data layout, so this kludge works. For now. For the first
		// test network.
		if( A->data_dim[1] != 0 || B->data_dim[1] != 0 ) {
			if( A->data_dim[0] == 1 )
				dimA = A->data_dim[1];
			else if( B->data_dim[0] == 1 )
				dimB = B->data_dim[1];
		}

		if( dimA != dimB )
			ERROR("Unimplemented or wrong input - input dimensions to Add do not match");

		Tensor *rv = new Tensor;
		rv->data_dim[0] = A->data_dim[0];
		rv->data_dim[1] = A->data_dim[1];
		rv->data_type = A->data_type;
		rv->data_num_elem = A->data_num_elem;
		outputs.push_back(rv);
	}
};
}
