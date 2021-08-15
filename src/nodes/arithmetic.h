/* This file is part of onnx2c.
 *
 * Generic Arithmetic node.
 * Some nodes are identical in description, differeing only in
 * the arithmetic operation.
 * Calculates elementvise C = A <op> B
 */
namespace toC {

class Arithmetic : public Node {
	public:
	Arithmetic(std::string op) {
		op_name = operand;
		A=B=C=NULL;

		if( op == "Div" )
			operand = "/";
		else if (op == "Add" )
			operand = "+";
		else if (op == "Mul" )
			operand = "*";
		else if (op == "Sub" )
			operand = "-";
		else
			ERROR("Arithmetic operand not implemented");

	}
	std::string operand;

	// input and output
	const Tensor *A;
	const Tensor *B;
	const Tensor *C;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		A->print_tensor_as_const(dst, !decorate);

		dst << ", ";
		B->print_tensor_as_const(dst, !decorate);

		dst << ", ";
		C->print_tensor(dst, !decorate);
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		ERROR("This should not have attributes!");
	}


	virtual void print(std::ostream &dst) const override
	{
		INDT_1 << "/* " << op_name << " */" << std::endl;

		// if either A or B does not have enough dimensions, prepend
		// dimensions of 1 to match rank of C
		std::vector<int> padA = A->data_dim;
		std::vector<int> padB = B->data_dim;
		for( unsigned i=0; i< (C->rank() - A->rank()); i++)
			padA.insert(padA.begin(), 0);
		for( unsigned i=0; i< (C->rank() - B->rank()); i++)
			padB.insert(padB.begin(), 0);

		// print out the loops over all C dimensions.
		// at the same time, create the indexing strings into A and B
		std::string Aidx = A->cname();
		std::string Bidx = B->cname();
		std::string Cidx = C->cname();
		for( unsigned r=0; r<C->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << C->data_dim[r] << "; " << lv << "++) {" << std::endl;

			if (padA[r]==1)
				Aidx += "[0]";
			else if(padA[r]!=0)
				Aidx += "[" + lv + "]";
			if (padB[r]==1)
				Bidx += "[0]";
			else if(padB[r]!=0)
				Bidx += "[" + lv + "]";
			Cidx +="[" + lv + "]";
		}


		if( options.quantize ) {
			INDT_2 << "int32_t tmp = " << Aidx << operand << Bidx << ";" << std::endl;
			// TODO: division amount here depends on operand
			INDT_2 << "tmp = tmp/2;" << std::endl;
			INDT_2 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_2 << "tmp = tmp < -127?-127:tmp;" << std::endl;
			INDT_2 << Cidx << "= tmp;" << std::endl;
		}
		else
			INDT_2 << Cidx << " = " << Aidx << operand << Bidx << ";" << std::endl;

		for( unsigned r=0; r<C->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		A = inputs[0];
		B = inputs[1];

		std::vector<int> result_dim;
		multidirectional_broadcast_size(A->data_dim, B->data_dim, result_dim);

		Tensor *t = new Tensor;
		t->data_dim = result_dim;
		t->data_type = A->data_type;
		C = t;
		outputs.push_back(t);
	}
};
}

