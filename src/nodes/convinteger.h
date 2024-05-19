/* This file is part of onnx2c.
 *
 * ConvInteger
 * Calculates an integer version of the convolution filter.
 *
 * Compared to the default Conv, the ConvInteger inputs (data
 * and weights both) are quantized with an offset. Presumably
 * this is to give better dynamic range for variables not centered
 * around zero.
 * These zero-point offsets are given as optional input tensors.
 */
#include "spatialfilter.h"
namespace toC {

class ConvInteger : public SpatialFilter {
	public:
	ConvInteger() {
		op_name = "ConvInteger";
		auto_pad = "NOTSET";
		group = 1;
	}

	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override
	{
		if( options.quantize )
			INDT_3 << "int32_t cell = 0;" << std::endl;
		else
			INDT_3 << "y[b][m][o0][o1] = 0;" << std::endl;
	}

	virtual void print_output_cell_calc(
		std::ostream &dst,
		const std::string &x_idx,
		const std::string &w_idx,
		const std::string &y_idx) const override
	{
		std::string x_zero="0";
		if( get_number_of_inputs() >= 3 ) // x_zero_point is optional, 3rd input
			x_zero = constant_acces_code( "x_zero_point[0]");

		std::string w_zero="0";
		if( get_number_of_inputs() >= 4 ) // w_zero_point is optional, 4th input
			w_zero = constant_acces_code( "w_zero_point[0]");


		INDT_4 << get_W()->data_type_str() << " w_ = " << constant_acces_code("w[m][c][k0][k1]") << ";" << std::endl;
		std::string dest;
		if( options.quantize )
			dest = "cell";
		else
			dest = "y[b][m][o0][o1]";

		INDT_4 << dest << "+= (x[b][c][i0+k0][i1+k1] - " << x_zero << ") * (w_ -" << w_zero << ");" << std::endl;
	}

	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const override
	{
		if( options.quantize ) {
			// TODO: this assumes 2D filter
			int divisor = kernel_shape[0]*kernel_shape[1]*16;
			INDT_3 << "int32_t tmp = cell/" << divisor << ";" << std::endl;
			INDT_3 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_3 << "tmp = tmp < -127?-127:tmp;" << std::endl;
			INDT_3 << "y[b][m][o0][o1] = tmp;" << std::endl;
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}

	virtual void resolve(void) override
	{
		name_input(0, "x");
		name_input(1, "w");

		if( get_number_of_inputs() > 2 )
			name_input(2, "x_zero_point");
		if( get_number_of_inputs() > 3 ){
			name_input(3, "w_zero_point");
		}

		if( get_X()->data_dim.size() != 4 )
			ERROR("Unimplemented: ConvInteger for non 2D images");


		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		if( group != 1 )
			ERROR("Unimplemented: ConvInteger: setting group to anything but 1");

		for( int d : dilations )
			if( d != 1 )
				ERROR("Unimplemented: ConvInteger: dilations other than 1");

		Tensor *rv = new Tensor;
		rv->data_dim = resolve_output_size();
		// ONNX specs say int32. local quantization is non conformant
		if( options.quantize )
			rv->data_type = onnx::TensorProto_DataType_INT8;
		else
			rv->data_type = onnx::TensorProto_DataType_INT32;
		register_output(rv, "y");
	}
};
}
