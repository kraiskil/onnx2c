/* This file is part of onnx2c.
 *
 * MatMulInteger
 * Matrix multiplication with integers.
 * In contrast to MatMul (which allows floats only)
 * MatMulInteger takes a input zero-point bias term
 * which is useful for quantized networks.
 */

namespace toC {

class MatMulInteger : public AbstractMatMul {
	public:
	MatMulInteger() {
		op_name = "MatMulInteger";
	}

	void print_multiply_accumulate(
		std::ostream &dst,
		const std::string &y_idx,
		const std::string &a_idx,
		const std::string &b_idx) const override {
		
		dst << y_idx << " += (" << a_idx << " - a_zero_point[0]) * ("
		    << b_idx << " - b_zero_point[0]);" << std::endl;
	}

	void resolve(void) override
	{
		Tensor *a = get_input_tensor(0);
		Tensor *b = get_input_tensor(1);

		name_input(0, "A");
		name_input(1, "B");

		if( get_number_of_inputs() > 2 ) {
			name_input(2, "a_zero_point");
			if ( get_input_tensor(2)->data_dim.size() != 1 ||
			     get_input_tensor(2)->data_dim[0] != 1 ) {
				ERROR("a_zero_point must be 1 dimensional with 1 element");
			}
		}

		if( get_number_of_inputs() > 3 ) {
			name_input(3, "b_zero_point");
			if ( get_input_tensor(3)->data_dim.size() != 1 ||
			     get_input_tensor(3)->data_dim[0] != 1 ) {
				ERROR("b_zero_point must be 1 dimensional with 1 element");
			}
		}

		Tensor *y = new Tensor;
		y->data_dim = resolve_shape(a, b);
		y->data_type = onnx::TensorProto_DataType_INT32;
		register_output(y, "Y");
	}
};

}
