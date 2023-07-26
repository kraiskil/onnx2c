/* This file is part of onnx2c.
 *
 * MatMulInteger
 * Matrix multiplication with integers.
 * In contrast to MatMul (which allows floats only)
 * MatMulInteger takes a input zero-point bias term
 * which is useful for quantized networks.
 *
 * TODO: share code with MatMul
 */

namespace toC {

class MatMulInteger : public Node {
	public:
	MatMulInteger() {
		op_name = "MatMulInteger";
	}

	virtual void print(std::ostream &dst) const override
	{
		Tensor *A = inputs[0];
		Tensor *B = inputs[1];
		Tensor *Y = outputs[0];
		std::string intype = A->data_type_str();
		std::string outtype = Y->data_type_str();
		std::string weighttype = B->data_type_str();
		std::string a_zero;
		std::string b_zero;

		if( A->data_dim.size() != 2 )
			ERROR("Unimplemented: higher than 2D MatMulInteger");

		int32_t rows = A->data_dim[0];
		int32_t cols = B->data_dim[1];
		int32_t inner = A->data_dim[1];
		int32_t inner2 = B->data_dim[0];
		if( inner == 0 ) inner=1;

		// TODO: handle the case of [N] * [Nx1] multiplication,
		//       i.e. shift rows to inner, set rows as 1
		//       and similarly, the case of input[1] being a 1D vector
		if( inner != inner2 )
			ERROR("MatMulInteger input's inner dimensions don't match");

		if( inputs.size() > 2)
			a_zero = "a_zero_point[0]";
		else
			a_zero = "0";
		if( inputs.size() > 3)
			b_zero = "b_zero_point[0]";
		else
			b_zero = "0";

		INDT_1 "/*MatMulInteger*/" << std::endl;
		INDT_1 << intype << " *A = (" << intype << "*)input_A;" << std::endl;
		INDT_1 << weighttype << " *B = (" << weighttype << "*)input_B;" << std::endl;
		INDT_1 << outtype << " *Y = (" << outtype << "*)output_Y;" << std::endl;

		INDT_1 << "for( uint32_t r=0; r<" << rows << "; r++ )" << std::endl;
		INDT_2 << "for( uint32_t c=0; c<" << cols << "; c++ ) {" << std::endl;


		// NB: quantization here is the experimental ONNXC quantization
		// that is not only integers, but also scales the output to 8bits.
		// This quantization terribly kludgy, and really should be removed
		if( options.quantize )
			INDT_3 << "int32_t sum = 0;" << std::endl;
		else
			INDT_3 << "Y[r*"<<cols<<" + c] = 0;" << std::endl;
		INDT_3 << "for( uint32_t i=0; i<" << inner << "; i++ )" << std::endl;
		if( options.quantize )
			INDT_4 << "sum";
		else
			INDT_4 << "Y[r*"<<cols<<"+c]";
		dst <<         "+= (A[r*"<<inner<< "+i] - " << a_zero << ")";
		dst <<           " * (B[i*"<<cols<<"+c] - " << b_zero << ");" << std::endl;

		if( options.quantize ) {
			INDT_3 << "int32_t tmp = sum/64;" << std::endl;
			INDT_3 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_3 << "tmp = tmp < -127?-127:tmp;" << std::endl;
			INDT_3 << "Y[r*"<<cols<<"+c] = tmp;" << std::endl;
		}

		INDT_2 "}" << std::endl;
	}

	virtual void resolve(void) override
	{
		register_input(inputs[0], "input_A");
		register_input(inputs[1], "input_B");

		if( inputs.size() > 2 ) {
			register_input(inputs[2], "a_zero_point");
			/* There is no backend reference test for this case */
			if( inputs[2]->data_dim[0] != 1 )
				ERROR("Unimplemented: 1D zero_point input");
		}
		if( inputs.size() > 3 ) {
			register_input(inputs[3], "b_zero_point");
			if( inputs[3]->data_dim[0] != 1 )
				ERROR("Unimplemented: 1D zero_point input");
		}

		int32_t rows, cols;
		result_dim(inputs, rows, cols);

		Tensor *rv = new Tensor;
		rv->data_dim.push_back(rows);
		rv->data_dim.push_back(cols);
		// ONNX specs say int32. local quantization is non conformant
		if( options.quantize )
			rv->data_type = onnx::TensorProto_DataType_INT8;
		else
			rv->data_type = onnx::TensorProto_DataType_INT32;
		register_output(rv, "output_Y");
	}

	void result_dim( const std::vector< Tensor*> &inputs, int32_t &rows, int32_t &cols) const
	{
		// TODO: this is the check for vectors. Check equivalent for N-dimensons: N>2
		if( inputs[0]->data_dim[1] != 0 && inputs[1]->data_dim[1] != 0 )
		{
			rows = inputs[0]->data_dim[0];
			cols = inputs[1]->data_dim[1];
		}
		else if( inputs[0]->data_dim[1] == 0 && inputs[1]->data_dim[1] == 0 )
		{
			ERROR("Bad input/unhandled: 2 vectors to MatMulInteger");
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
