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
		x=w=x_zero_point=w_zero_point=y=NULL;
	}
	/* ConvInteger node specific attributes */

	// optional inputs
	const Tensor *x_zero_point;
	const Tensor *w_zero_point;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		x->print_tensor(dst, !decorate);
		dst << ", ";
		w->print_tensor(dst, !decorate);
		dst << ", ";
		if( x_zero_point ) {
			x_zero_point->print_tensor(dst, !decorate);
			dst << ", ";
		}
		if( w_zero_point ) {
			w_zero_point->print_tensor(dst, !decorate);
			dst << ", ";
		}
		y->print_tensor(dst, !decorate);
	}

	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const
	{
		if( quantize )
			INDT_3 << "int32_t cell = 0;" << std::endl;
		else
			INDT_3 << y->cname() << "[b][m][o0][o1] = 0;" << std::endl;
	}
	virtual void print_output_cell_calc(std::ostream &dst, const std::string &x_idx, const std::string &w_idx, const std::string &y_idx) const
	{
		std::string x_zero;
		if( x_zero_point )
			x_zero = constant_acces_code( x_zero_point->cname() + "[0]");
		else
			x_zero = "0";

		INDT_4 << w->data_type_str() << " w = " << constant_acces_code( w->cname() + "[m][c][k0][k1]") << ";" << std::endl;
		std::string dest;
		if( quantize )
			dest = "cell";
		else
			dest = y->cname() + "[b][m][o0][o1]";

		INDT_4 << dest << "+= ("<< x->cname() << "[b][c][i0+k0][i1+k1] - " << x_zero << ") * w;" << std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const
	{
		if( quantize ) {
			// TODO: this assumes 2D filter
			int divisor = kernel_shape[0]*kernel_shape[1]*16;
			INDT_3 << "int32_t tmp = cell/" << divisor << ";" << std::endl;
			INDT_3 << "tmp = tmp > 127?127:tmp;" << std::endl;
			INDT_3 << "tmp = tmp < -127?-127:tmp;" << std::endl;
			INDT_3 << y->cname() + "[b][m][o0][o1] = tmp;" << std::endl;
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		x = inputs[0]; // data
		w = inputs[1]; // weights

		if( inputs.size() > 2 )
			x_zero_point = inputs[2];
		if( inputs.size() > 3 ){
			w_zero_point = inputs[3];
			ERROR("unimplemented: weight zero points");
		}

		if( x->data_dim.size() != 4 )
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
		if( quantize )
			rv->data_type = onnx::TensorProto_DataType_INT8;
		else
			rv->data_type = onnx::TensorProto_DataType_INT32;
		y=rv;
		outputs.push_back(rv);
	}
};
}
