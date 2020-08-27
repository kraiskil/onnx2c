#include "error.h"

namespace toC {

class Relu : public Node {
	public:
	Relu() {
		op_name = "Relu";
	}

	virtual void print(std::ostream &dst) const
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Relu");
		if( outputs.size() != 1 )
			ERROR("wrong number of outputs from Relu");
		std::string type = inputs[0]->data_type_str();

		dst << "\t/*Relu*/" << std::endl;
		
		dst << "\t" << type << " *X = (" << type << "*)" << inputs[0]->cname() << ";" << std::endl;
		dst << "\t" << type << " *Y = (" << type << "*)" << outputs[0]->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << inputs[0]->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\tY[i] = X[i] > 0 ? X[i] : 0;" << std::endl;
		dst << std::endl;
	} 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		const Tensor *A = inputs[0];
		if(  typeConstraint_allFloatingPoints(A) == false )
			ERROR("Incorrect input for Relu"); 

		if( A->data_dim[1] != 0 && !(A->data_dim[0]!=1 || A->data_dim[1]!=1) )
			ERROR("Unimplemented - multidimiensional Relu");

		Tensor *rv = new Tensor;
		for( auto d : A->data_dim )
			rv->data_dim.push_back(d);
		rv->data_type = A->data_type;
		outputs.push_back(rv);
	}
};
}
