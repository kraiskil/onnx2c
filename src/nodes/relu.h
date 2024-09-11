#include "error.h"

namespace toC {

class Relu : public Node {
	public:
	Relu() {
		op_name = "Relu";
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *X=get_input_tensor(0);
		std::string type = X->data_type_str();

		dst << "\t/*Relu*/" << std::endl;
		dst << "\t" << type << " *X_ptr = (" << type << "*)X;" << std::endl;
		dst << "\t" << type << " *Y_ptr = (" << type << "*)Y;" << std::endl;
		dst << "\tuint32_t i;" << std::endl;

		dst << "\t" << "for( i=0; i<" << X->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\tY_ptr[i] = X_ptr[i] > 0 ? X_ptr[i] : 0;" << std::endl;
		dst << std::endl;
	} 

	virtual void resolve(void) override
	{
		const Tensor *X = get_input_tensor(0);
		name_input(0, "X");
		if((  typeConstraint_allFloatingPoints(X)
		    ||typeConstraint_signed_integers(X)   ) == false )
			ERROR("Incorrect input for Relu"); 

		if( X->data_dim[1] != 0 && !(X->data_dim[0]!=1 || X->data_dim[1]!=1) )
			ERROR("Unimplemented - multidimiensional Relu");

		Tensor *rv = new Tensor;
		for( auto d : X->data_dim )
			rv->data_dim.push_back(d);
		rv->data_type = X->data_type;
		register_output(rv, "Y");
	}
};
}
