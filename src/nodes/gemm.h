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
		alpha=beta=1;
		transA=transB=0;
	}

	/* Node attributes */
	float alpha;
	float beta;
	int transA; // boolean for 'do the tranpose'
	int transB;

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
		const Tensor *A  = inputs[0];
		const Tensor *B  = inputs[1];
		const Tensor *C  = inputs.size() > 2 ? inputs[2]:nullptr;
		//int A1 = A->data_dim[1];
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
		dst << "\t */" << std::endl;

		// Helper variables to make the code (both this and generated) cleaner
		dst << "\t" << "const int M = " << M << ";" << std::endl;
		dst << "\t" << "const int K = " << K << ";" << std::endl;
		dst << "\t" << "const int N = " << N << ";" << std::endl;
		dst << "\t" << "float alpha = " << alpha << ";" << std::endl;
		dst << "\t" << "float beta = " << beta << ";" << std::endl;

		std::string A_el = transA ? "A[i][r]" : "A[r][i]";
		std::string B_idx = transB ? "[c][i]" : "[i][c]";

		// Cast optional C matrix to generated variable
		// "C_[M][N]"
		std::string C_idx;
		if( C  ) {
			C_idx = "";
			int dim;
			switch (C->rank())
			{
				case 0:
					ERROR("Unimplemented: scalar C in Gemm");
					break;
				case 1:
					dim = C->data_dim[0];
					if( dim == M ){
						C0=M;
						C1=1;
					}
					else if ( dim == N ) {
						C0=1;
						C1=N;
					}
					else if ( dim == 1 ) {
						C0=1;
						C1=1;
					}
					else {
						ERROR("C dimension mismatch in Gemm");
					}
					break;
				case 2:
					C0=C->data_dim[0];
					C1=C->data_dim[1];
					break;
				default:
					ERROR("C has too many dimensions in Gemm");
			}
			if( C0 <= 1 )
				C_idx += "[0]";
			else
				C_idx += "[r]";
			if( C1 <= 1 )
				C_idx += "[0]";
			else
				C_idx += "[c]";
			INDT_1 << type << " (*C_)["<<C1<<"]  = (" << type << "(*)["<<C1<<"])C;" << std::endl;
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
		INDT_4 <<   B->data_type_str() << " B_el = " << constant_acces_code( "B" + B_idx ) << ";" << std::endl;
		INDT_4 <<   "ABrc += " << A_el << " * B_el;" << std::endl;
		INDT_3 << "}" << std::endl;


		/* Add scale & bias, store result in output */
		if( options.quantize )
			INDT_3 << "int32_t tmp = ABrc * alpha;" << std::endl;
		else
			INDT_3 << type <<" tmp = ABrc * alpha;" << std::endl;

		if( C ) {
			INDT_3 << "tmp += C_" << C_idx << " * beta;" << std::endl;
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
	virtual void resolve(void) override
	{
		if (inputs.size() < 2)
			ERROR("Not enough inputs");

		const Tensor *A  = inputs[0];
		const Tensor *B  = inputs[1];
		register_input(A, "A");
		register_input(B, "B");

		if (inputs.size() == 3) {
			register_input(inputs[2], "C");
		}

		// output dimensions - see the specification
		int M = transA ? A->data_dim[1] : A->data_dim[0];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor *t = new Tensor;
		t->data_dim.push_back(M);
		t->data_dim.push_back(N);
		t->data_type = A->data_type;
		register_output(t, "Y");
	}
};
}

