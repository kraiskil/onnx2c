/* This file is part of onnx2c.
 *
 * Generic Elementwise applied functions node.
 * Some nodes are identical in description, differeing only in
 * the function applied
 * Calculates elementwise Y = func ( A )
 */
namespace toC {

class Elementwise : public Node {
	public:
	Elementwise(std::string op) {
		op_name = operand;
		X=Y=NULL;

		if( op == "Abs" )
			operand = "fabs";
		else if( op == "Ceil" )
			operand = "ceil";
		else if( op == "Floor" )
			operand = "floor";
		else
			ERROR("Elementwise operand not implemented");

	}
	std::string operand;

	const Tensor *X;
	const Tensor *Y;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor(dst, !decorate);

		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		ERROR("This should not have attributes!");
	}


	virtual void print(std::ostream &dst) const override
	{
		INDT_1 << "/* " << op_name << " */" << std::endl;

		// print out the loops over all C dimensions.
		// at the same time, create the indexing strings into X and Y
		std::string Xidx = X->cname();
		std::string Yidx = Y->cname();
		for( unsigned r=0; r< Y->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << Y->data_dim[r] << "; " << lv << "++) {" << std::endl;

			Xidx += "[" + lv + "]";
			Yidx += "[" + lv + "]";
		}

		INDT_2 << Yidx << " = " << operand << "( " << Xidx << " );" << std::endl;

		for( unsigned r=0; r<Y->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		X = inputs[0];

		Tensor *t = new Tensor;
		t->data_dim = X->data_dim;
		t->data_type = X->data_type;
		Y = t;
		outputs.push_back(t);
	}
};
}

