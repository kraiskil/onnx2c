/* This file is part of onnx2c.
 *
 * AveragePool
 * Calculates average from the elements that
 * are under the kernel.
 *
 */
#include <cmath>
#include "pooling.h"
namespace toC {

class AveragePool : public Pooling {
	public:
	AveragePool() : Pooling() {
		op_name = "AveragePool";
		Indices = NULL;
	}

	// optional outputs
	const Tensor *Indices;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		x->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		y->print_tensor(dst, !decorate);
		if( Indices->name != "" ) {
			dst << ", ";
			Indices->print_tensor(dst, !decorate);
		}
	}



	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override
	{
		INDT_3  << y->data_type_str() << " curavg = 0.0;" << std::endl;
		INDT_3  << "int numavg = 0;" << std::endl;
	}
	virtual void print_output_cell_calc(
		std::ostream &dst,
		const std::string &x_idx,
		const std::string &w_idx,
		const std::string &y_idx) const override
	{
		// Sum up the cells
		INDT_4 << "numavg += 1;" <<std::endl;
		INDT_4 << "curavg += " << x->cname() << x_idx << ";" <<std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const override
	{
		// Calculate the averageing part
		if( count_include_pad ) {
			int numavg=1;
			for ( auto k : kernel_shape )
				numavg *= k;
			INDT_3 << "/* Counting padding into the average is requested */" << std::endl;
			INDT_3 << "numavg = " << numavg << ";" << std::endl;
		}
		INDT_3 << y->cname() << y_idx << "= curavg/numavg;" << std::endl;
	}


	virtual void print(std::ostream &dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}
 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		x = inputs[0];

		if( !(  typeConstraint_plainFloatingPoints(x)
		      ||typeConstraint_8bit(x)) )
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

		update_pads();

		// optional indices vector
		Tensor *indices_out = new Tensor;
		indices_out->data_type = onnx::TensorProto_DataType::TensorProto_DataType_INT64;
		indices_out->data_dim = rv->data_dim;
		Indices = indices_out;
		outputs.push_back(indices_out);
	}
};
}
