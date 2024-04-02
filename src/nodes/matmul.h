
namespace toC {

class MatMul : public Node {
	public:
	MatMul() {
		op_name = "MatMul";
	}

	std::string vecstr( const std::vector<int>& vec ) const
	{
		std::stringstream result;
		result << "{ ";
		std::copy( vec.begin(), vec.end(), std::ostream_iterator<int>( result, ", " ) );
		result << "}";
		return result.str();
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *A = get_input_tensor(0);
		const Tensor *B = get_input_tensor(1);
		std::string type = A->data_type_str();

		bool A_is_correct_size = A->data_dim.size() == 2 || A->data_dim.size() == 3;
		bool B_is_correct_size = B->data_dim.size() == 2 || B->data_dim.size() == 3;
		if ( !A_is_correct_size || !B_is_correct_size )
		{
			ERROR( std::string( "Unimplemented: MatMul with dimensions: A: " ) + vecstr( A->data_dim ) + ", B: " + vecstr( B->data_dim ) );
		}

		std::vector<int> A_dim( A->data_dim.begin() + A->data_dim.size() - 2, A->data_dim.end() );
		std::vector<int> B_dim( B->data_dim.begin() + B->data_dim.size() - 2, B->data_dim.end() );

		int32_t rows = A_dim[0];
		int32_t cols = B_dim[1];
		int32_t inner = A_dim[1];
		int32_t inner2 = B_dim[0];
		if( inner == 0 ) inner=1;

		// TODO: handle the case of [N] * [Nx1] multiplication,
		//       i.e. shift rows to inner, set rows as 1
		//       and similarly, the case of input[1] being a 1D vector 
		if( inner != inner2 )
			ERROR("MatMul input's inner dimensions don't match");

		bool A_is_2d = A->data_dim.size() == 2;
		bool B_is_2d = B->data_dim.size() == 2;

		if ( A_is_2d && B_is_2d )
		{
			INDT_1 << "/* MatMul */" << std::endl;
			INDT_1 << "for( uint32_t r=0; r<" << rows << "; r++ )" << std::endl;
			INDT_2 << "for( uint32_t c=0; c<" << cols << "; c++ ) {" << std::endl;
			INDT_3 << "Y[r][c] = 0;" << std::endl;
			INDT_3 << "for( uint32_t i=0; i<" << inner << "; i++ )" << std::endl;
			INDT_4 << "Y[r][c] += A[r][i] * B[i][c];" << std::endl;
			INDT_2 << "}" << std::endl;
		}
		else
		{
			std::string A_txt = A_is_2d ? "A" : "A[n]";
			std::string B_txt = B_is_2d ? "B" : "B[n]";

			INDT_1 << "/* MatMul */" << std::endl;

			INDT_1 << "for( uint32_t n=0; n<" << A->data_dim[0] << "; n++ ) {" << std::endl;

			INDT_2 << "for( uint32_t r=0; r<" << rows << "; r++ )" << std::endl;
			INDT_3 << "for( uint32_t c=0; c<" << cols << "; c++ ) {" << std::endl;
			INDT_4 << "Y[n][r][c] = 0;" << std::endl;
			INDT_4 << "for( uint32_t i=0; i<" << inner << "; i++ )" << std::endl;
			INDT_5 << "Y[n][r][c] += " << A_txt << "[r][i] * " << B_txt << "[i][c];" << std::endl;
			INDT_3 << "}" << std::endl;

			INDT_1 << "}" << std::endl;
		}
	} 
	virtual void resolve(void) override
	{
		const Tensor *A = get_input_tensor(0);
		const Tensor *B = get_input_tensor(1);
		name_input(0, "A");
		name_input(1, "B");
		if(  typeConstraint_highPrecisionNumeric(A) == false )
			ERROR("Incorrect input for MatMul"); 
		if(  typeConstraint_highPrecisionNumeric(B) == false )
			ERROR("Incorrect input for MatMul"); 

		int32_t rows, cols;
		result_dim(rows, cols);

		Tensor *rv = new Tensor;

		if ( A->data_dim.size() == 3 && B->data_dim.size() == 3 )
		{
			if ( A->data_dim[0] != B->data_dim[0] )
				ERROR( std::string("MatMul input's dimensions don't match: A: ") + vecstr( A->data_dim ) + ", B: " + vecstr( B->data_dim ) );

			rv->data_dim.push_back( A->data_dim[0] );
		}
		else if ( A->data_dim.size() == 3 && B->data_dim.size() == 2 )
		{
			rv->data_dim.push_back( A->data_dim[0] );
		}
		else if ( A->data_dim.size() == 2 && B->data_dim.size() == 3 )
		{
			rv->data_dim.push_back( B->data_dim[0] );
		}

		rv->data_dim.push_back(rows);
		rv->data_dim.push_back(cols);
		rv->data_type = A->data_type;
		register_output(rv, "Y");
	}

	void result_dim( int32_t &rows, int32_t &cols) const
	{
		const Tensor* A = get_input_tensor( 0 );
		const Tensor* B = get_input_tensor( 1 );

		std::vector<int> A_dim( A->data_dim.begin() + A->data_dim.size() - 2, A->data_dim.end() );
		std::vector<int> B_dim( B->data_dim.begin() + B->data_dim.size() - 2, B->data_dim.end() );

		// TODO: this is the check for vectors. Check equivalent for N-dimensons: N>2
		if ( A_dim[1] != 0 && B_dim[1] != 0 )
		{
			rows = A_dim[0];
			cols = B_dim[1];
		}
		else if ( A_dim[1] == 0 && B_dim[1] == 0 )
		{
			ERROR( "Bad input/unhandled: 2 vectors to MatMul" );
		}
		else if ( A_dim[1] == 0 )
		{
			cols = B_dim[1];
			if ( A_dim[0] == B_dim[0] )
				rows = 1;
			else
				rows = A_dim[0];
		}
		else
		{
			rows = A_dim[0];
			if ( A_dim[1] == B_dim[0] )
				cols = 1;
			else
				cols = B_dim[0];
		}
	}
};
}
