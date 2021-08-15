/* This file is part of onnx2c.
 *
 * "GEneral Matrix Multiplication"
 * Calulates:
 * Y = alpha*A*B + beta*C
 * optionally trasposing A and/or B first.
 * C need not be of size A*B, but must be
 * 'unidirectionally broadcastable' to A*B.
 */
namespace toC {

class Gemm : public Node {
	public:
	Gemm() {
		op_name = "Gemm";
		A=B=C=Y=NULL;
		alpha=beta=1;
		transA=transB=0;
	}

	/* Node attributes */
	float alpha;
	float beta;
	int transA; // boolean for 'do the tranpose'
	int transB;

	// input and output
	const Tensor *A;
	const Tensor *B;
	const Tensor *C; // optional
	const Tensor *Y;


	/* Print the function parameters - use the order they are introduced in the
	 * ONNX documentation */
	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		A->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		B->print_tensor_as_const(dst, !decorate);

		if (C) {
			dst << ", ";
			C->print_tensor_as_const(dst, !decorate);
		}

		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}

	/* Parse attributes, if this node has them. */
	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;

			if( a.name() == "alpha" )
				alpha = parse_attribute_float(a);
			else if ( a.name() == "beta" )
				beta = parse_attribute_float(a);
			else if ( a.name() == "transA" )
				transA = parse_attribute_int(a);
			else if ( a.name() == "transB" )
				transB = parse_attribute_int(a);
			else
				ERROR("unknown attribute: " << a.name());
		}
	}



	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		int A0 = A->data_dim[0];
		int A1 = A->data_dim[1];
		int C0,C1; C0=C1=0;
		if( C ) {
			C0 = C->data_dim[0];
			if ( C->rank() > 1 ) {
				C1 = C->data_dim[1];
			}
		}
	
		int M = transA ? A->data_dim[1] : A->data_dim[0]; // row
		int K = transA ? A->data_dim[0] : A->data_dim[1]; // inner
		int N = transB ? B->data_dim[0] : B->data_dim[1]; // column
		std::string type = A->data_type_str();

		// Documentation if someone is reading the code
		dst << "\t/* Gemm */" << std::endl;
		dst << "\t/* alpha   = " << alpha << std::endl;
		dst << "\t   beta    = " << beta << std::endl;
		dst << "\t   transA  = " << transA << std::endl;
		dst << "\t   transB  = " << transB << std::endl;
		dst << "\t   A       = ("<< A0 << "x"<< A1 << ")"<< std::endl;
		dst << "\t   datatype= "<< type << std::endl;
		dst << "\t */" << std::endl;

		// Helper variables to make the code (both this and generated) cleaner
		dst << "\t" << "const int M = " << M << ";" << std::endl;
		dst << "\t" << "const int K = " << K << ";" << std::endl;
		dst << "\t" << "const int N = " << N << ";" << std::endl;
		// C has some ugly syntax sometimes, but now we can do A[r][c]
		dst << "\t" << type << " (*A)["<<A1<<"]  = (" << type << "(*)["<<A1<<"])" << A->cname() << ";" << std::endl;
		dst << "\t" << type << " (*Y)["<<N<<"]  = (" << type << "(*)["<<N<<"])" << Y->cname() << ";" << std::endl;
		dst << "\t" << "float alpha = " << alpha << ";" << std::endl;
		dst << "\t" << "float beta = " << beta << ";" << std::endl;

		std::string A_el = transA ? "A[i][r]" : "A[r][i]";
		std::string B_idx = transB ? "[c][i]" : "[i][c]";
		std::string C_idx;
		if( C  ) {
			C_idx = "";
			if( C0 == 1 )
				C_idx += "[0]";
			else
				C_idx += "[r]";
			if( C->rank() > 1 ) {
				if( C1 == 1 )
					C_idx += "[0]";
				else
					C_idx += "[c]";
			}
		}


		// Now genereate the calculation source code

		// Loop output rows, columns
		INDT_1 << "for( uint32_t r=0; r<M; r++ )" << std::endl;
		INDT_2 << "for( uint32_t c=0; c<N; c++ ) {" << std::endl;

		/* Calculate the matrix muliplication dot inner dot product */
		if( options.quantize ) {
			INDT_3 << "int32_t ABrc = 0;" << std::endl;
		}
		else {
			INDT_3 << type <<" ABrc = 0;" << std::endl;
		}
		INDT_3 << "for( uint32_t i=0; i<K; i++ ) {" << std::endl;
		INDT_4 <<   B->data_type_str() << " B = " << constant_acces_code( B->cname() + B_idx ) << ";" << std::endl;
		INDT_4 <<   "ABrc += " << A_el << " * B;" << std::endl;
		INDT_3 << "}" << std::endl;


		/* Add scale & bias, store result in output */
		if( options.quantize )
			INDT_3 << "int32_t tmp = ABrc * alpha;" << std::endl;
		else
			INDT_3 << type <<" tmp = ABrc * alpha;" << std::endl;

		if( C ) {
			INDT_3 << C->data_type_str() << " C = " << constant_acces_code( C->cname() + C_idx ) << ";" << std::endl;
			INDT_3 << "tmp += C * beta;" << std::endl;
		}

		if( options.quantize ) {
			INDT_3 << "tmp = tmp/(K*16);" << std::endl;
			INDT_3 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_3 << "tmp = tmp < -127?-127:tmp;" << std::endl;
		}

		INDT_3 << "Y[r][c] = tmp;" << std::endl;


		INDT_1 << "}" << std::endl;
	}


	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if (inputs.size() < 2)
			ERROR("Not enough inputs");

		A  = inputs[0];
		B  = inputs[1];

		if (inputs.size() == 3)
			C = inputs[2];

		// output dimensions - see the specification
		int M = transA ? A->data_dim[1] : A->data_dim[0];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor *t = new Tensor;
		t->data_dim.push_back(M);
		t->data_dim.push_back(N);
		t->data_type = A->data_type;
		/* Store the created tensor both as reference in this node, and into
		 * the return value vector! */
		outputs.push_back(t);
		Y = t;
	}
};
}

