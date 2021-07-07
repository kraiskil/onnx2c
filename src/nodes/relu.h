#include "error.h"

namespace toC {

class Relu : public Node {
	public:
	Relu() {
		op_name = "Relu";
		X=Y=NULL;
	}
	// inputs
	const Tensor *X;
	// outputs
	const Tensor *Y;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{
		std::string type = X->data_type_str();

		dst << "\t/*Relu*/" << std::endl;
		
		dst << "\t" << type << " *X = (" << type << "*)" << X->cname() << ";" << std::endl;
		dst << "\t" << type << " *Y = (" << type << "*)" << Y->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << X->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\tY[i] = X[i] > 0 ? X[i] : 0;" << std::endl;
		dst << std::endl;
	} 

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		X = inputs[0];
		if((  typeConstraint_allFloatingPoints(X)
		    ||typeConstraint_signed_integers(X)   ) == false )
			ERROR("Incorrect input for Relu"); 

		if( X->data_dim[1] != 0 && !(X->data_dim[0]!=1 || X->data_dim[1]!=1) )
			ERROR("Unimplemented - multidimiensional Relu");

		Tensor *rv = new Tensor;
		for( auto d : X->data_dim )
			rv->data_dim.push_back(d);
		rv->data_type = X->data_type;
		Y=rv;
		outputs.push_back(rv);
	}
};
}
