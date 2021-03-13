/* This file is part of onnx2c.
 *
 * Conv
 * Calculates an "industry standard" convolution filter.
 */

#include "spatialfilter.h"
namespace toC {

class Conv : public SpatialFilter {
	public:
	Conv() {
		op_name = "Conv";
		auto_pad = "NOTSET";
		group = 1;
		b=NULL;
	}
	/* Conv node specific attributes */

	// optional inputs
	const Tensor *b;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		x->print_tensor(dst, !decorate);
		dst << ", ";
		w->print_tensor(dst, !decorate);
		dst << ", ";
		if( b ) {
			b->print_tensor(dst, !decorate);
			dst << ", ";
		}
		y->print_tensor(dst, !decorate);
	}


	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const
	{
		INDT_3 << y->cname() << "[b][m][o0][o1] = ";
		if( b == NULL )
			dst << "0;" << std::endl;
		else
			dst << b->cname() << "[m];" << std::endl;
	};
	virtual void print_output_cell_calc(std::ostream &dst, const std::string &x_idx, const std::string &w_idx, const std::string &y_idx) const
	{
		INDT_4 << y->cname() << "[b][m][o0][o1] += "<< x->cname() << "[b][c][i0+k0][i1+k1] *";
		   dst <<             w->cname() << "[m][c][k0][k1];" << std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const
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
		if( inputs.size() == 3 ) {
			b = inputs[2];
		}
		else
			b = NULL;

		if(  typeConstraint_highPrecisionNumeric(x) == false
		   ||typeConstraint_highPrecisionNumeric(w) == false)
			ERROR("Incorrect input for node");
		if( b && (typeConstraint_highPrecisionNumeric(b) == false) )
			ERROR("Incorrect input for node");

		if( x->data_dim.size() != 4 )
			ERROR("Unimplemented: Conv for non 2D images");


		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		if( group != 1 )
			ERROR("Unimplemented: Conv: setting group to anything but 1");

		for( int d : dilations )
			if( d != 1 )
				ERROR("Unimplemented: Conv: dilations other than 1");

		Tensor *rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = x->data_type;
		y=rv;
		outputs.push_back(rv);
	}
};
}
