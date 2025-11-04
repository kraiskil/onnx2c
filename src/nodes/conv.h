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
	}

	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override
	{
		INDT_3 << "y" << y_idx << " = ";
		if( get_number_of_inputs() < 3 ) // bias is the 3rd input, optional
			dst << "0;" << std::endl;
		else
			dst << "bias[m];" << std::endl;
	};
	virtual void print_output_cell_calc(
		std::ostream &dst,
		const std::string &x_idx,
		const std::string &w_idx,
		const std::string &y_idx) const override
	{
		std::string kidx="";
		for(unsigned i=0; i<get_numDataDim(); i++){
			kidx+= "[k" + std::to_string(i) + "]";
		}
		INDT_4 << "y" << y_idx << " += x" << x_idx << " *";
		if( group == 1 )
		   dst << "w[m][c]"<<kidx<<";" << std::endl;
		else
		   dst << "w[m][c-(gi*g)]"<<kidx<<";" << std::endl;
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
		name_input(0,"x");
		name_input(1,"w");
		if( get_number_of_inputs() == 3 ) {
			name_input(2,"bias");
		}

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		Tensor *rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = get_X()->data_type;
		register_output(rv, "y");
	}
};
}
