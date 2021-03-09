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
#include "convolutions.h"
namespace toC {

class ConvInteger : public Convolutions {
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

	virtual void print_output_cell_init(std::ostream &dst) const
	{
		INDT_3 << y->cname() << "[b][m][o0][o1] = 0;" << std::endl;
	}
	virtual void print_output_cell_calc(std::ostream &dst) const
	{
		std::string x_zero;
		if( x_zero_point )
			x_zero = x_zero_point->cname() + "[0]";
		else
			x_zero = "0";

		INDT_4 << y->cname() << "[b][m][o0][o1] += ("<< x->cname() << "[b][c][i0+k0][i1+k1] - " << x_zero << ") *";
		   dst <<                w->cname() << "[m][c][k0][k1];" << std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst) const
	{
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

		if(  typeConstraint_8bit(x) == false
		   ||typeConstraint_8bit(w) == false)
			ERROR("Incorrect input for node");

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
		rv->data_type = onnx::TensorProto_DataType_INT32;
		y=rv;
		outputs.push_back(rv);
	}
};
}
