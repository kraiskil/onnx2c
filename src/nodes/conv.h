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
		x->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		w->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		if( b ) {
			b->print_tensor_as_const(dst, !decorate);
			dst << ", ";
		}
		y->print_tensor(dst, !decorate);
	}


	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override
	{
		std::string outidx="";
		for(unsigned i=0; i<x->rank()-2; i++)
			outidx += "[o" + std::to_string(i) + "]";
		INDT_3 << y->cname() << "[b][m]" << outidx << " = ";
		if( b == NULL )
			dst << "0;" << std::endl;
		else
			dst << b->cname() << "[m];" << std::endl;
	};
	virtual void print_output_cell_calc(
		std::ostream &dst,
		const std::string &x_idx,
		const std::string &w_idx,
		const std::string &y_idx) const override
	{
		std::string outidx="";
		std::string iididx="";
		std::string kidx="";
		for(unsigned i=0; i<x->rank()-2; i++){
			outidx += "[o" + std::to_string(i) + "]";
			iididx+= "[ii" + std::to_string(i) + "]";
			kidx+= "[k" + std::to_string(i) + "]";
		}
		INDT_4 << y->cname() << "[b][m]"<<outidx<<" += "<< x->cname() << "[b][c]"<<iididx<<" *";
		if( group == 1 )
		   dst <<             w->cname() << "[m][c]"<<kidx<<";" << std::endl;
		else
		   dst <<             w->cname() << "[m][c-(gi*g)]"<<kidx<<";" << std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const override
	{
	}
	virtual void print(std::ostream &dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}
 
	virtual void resolve(void) override
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

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		Tensor *rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = x->data_type;
		y=rv;
		outputs.push_back(rv);
	}
};
}
