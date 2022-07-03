
namespace toC {

class MatMul : public Node {
	public:
	MatMul() {
		op_name = "MatMul";
		A=B=Y=NULL;
	}
	// inputs
	const Tensor *A;
	const Tensor *B;
	// outputs
	const Tensor *Y;


	virtual void print(std::ostream &dst) const override
	{
		std::string type = A->data_type_str();

		if( A->data_dim.size() != 2 )
			ERROR("Unimplemented: higher than 2D MatMul");

		int32_t rows = A->data_dim[0];
		int32_t cols = B->data_dim[1];
		int32_t inner = A->data_dim[1];
		int32_t inner2 = B->data_dim[0];
		if( inner == 0 ) inner=1;

		// TODO: handle the case of [N] * [Nx1] multiplication,
		//       i.e. shift rows to inner, set rows as 1
		//       and similarly, the case of input[1] being a 1D vector 
		if( inner != inner2 )
			ERROR("MatMul input's inner dimensions don't match");

		INDT_1 << "/* MatMul */" << std::endl;
		INDT_1 << "for( uint32_t r=0; r<" << rows << "; r++ )" << std::endl;
		INDT_2 <<   "for( uint32_t c=0; c<" << cols << "; c++ ) {" << std::endl;
		INDT_3 <<     "Y[r][c] = 0;" << std::endl;
		INDT_3 <<     "for( uint32_t i=0; i<" << inner << "; i++ )" << std::endl;
		INDT_4 <<        "Y[r][c] += A[r][i] * B[i][c];" << std::endl;
		INDT_2 <<   "}" << std::endl;

	} 
	virtual void resolve(void) override
	{
		A = inputs[0];
		B = inputs[1];
		register_input(A, "A");
		register_input(B, "B");
		if(  typeConstraint_highPrecisionNumeric(A) == false )
			ERROR("Incorrect input for MatMul"); 
		if(  typeConstraint_highPrecisionNumeric(B) == false )
			ERROR("Incorrect input for MatMul"); 

		int32_t rows, cols;
		result_dim(inputs, rows, cols);
	

		Tensor *rv = new Tensor;
		rv->data_dim.push_back(rows);
		rv->data_dim.push_back(cols);
		rv->data_type = A->data_type;
		Y=rv;
		register_output(rv, "Y");
	}

	void result_dim( const std::vector< const Tensor*> &inputs, int32_t &rows, int32_t &cols) const
	{
		// TODO: this is the check for vectors. Check equivalent for N-dimensons: N>2
		if( inputs[0]->data_dim[1] != 0 && inputs[1]->data_dim[1] != 0 )
		{
			rows = inputs[0]->data_dim[0];
			cols = inputs[1]->data_dim[1];
		}
		else if( inputs[0]->data_dim[1] == 0 && inputs[1]->data_dim[1] == 0 )
		{
			ERROR("Bad input/unhandled: 2 vectors to MatMul");
		}
		else if( inputs[0]->data_dim[1] == 0 )
		{
			cols = inputs[1]->data_dim[1];
			if( inputs[0]->data_dim[0] == inputs[1]->data_dim[0] )
				rows = 1;
			else
				rows = inputs[0]->data_dim[0];
		}
		else
		{
			rows = inputs[0]->data_dim[0];
			if( inputs[0]->data_dim[1] == inputs[1]->data_dim[0] )
				cols = 1;
			else
				cols = inputs[1]->data_dim[0];
		}
	}
};
}
