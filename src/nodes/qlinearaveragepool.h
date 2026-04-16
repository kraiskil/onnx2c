/* This file is part of onnx2c.
 *
 * QLinearAveragePool
 *
 * Quantized average pooling (QOperator form).
 *
 * Computes:
 *   y = round( (x_scale/(K*y_scale)) * sum(x - x_zero_point) ) + y_zero_point
 * with int64 accumulation.
 */

#pragma once

#include "pooling.h"
#include <cmath>

namespace toC {

class QLinearAveragePool : public Pooling {
	public:
	QLinearAveragePool() : Pooling()
	{
		op_name = "QLinearAveragePool";
	}

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		const Tensor* t = get_input_tensor(input_no);
		if (!(t->data_dim.size() == 0 || (t->data_dim.size() == 1 && t->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void print_output_cell_init(std::ostream& dst, const std::string&) const override
	{
		INDT_3 << "int64_t sum = 0;" << std::endl;
		INDT_3 << "int32_t num = 0;" << std::endl;
	}

	void print_output_cell_calc(std::ostream& dst,
	                            const std::string& x_idx,
	                            const std::string&,
	                            const std::string&) const override
	{
		INDT_4 << "num += 1;" << std::endl;
		INDT_4 << "sum += (int64_t)((int32_t)x" << x_idx << " - (int32_t)x_zero_point[0]);" << std::endl;
	}

	void print_output_cell_finalize(std::ostream& dst, const std::string& y_idx) const override
	{
		const Tensor* y = get_Y();
		auto [lower, upper] = y->get_type_bounds();

		if (count_include_pad) {
			int64_t kcount = 1;
			for (auto k : kernel_shape)
				kcount *= k;
			INDT_3 << "/* Counting padding into the average is requested */" << std::endl;
			INDT_3 << "num = " << kcount << ";" << std::endl;
		}

		std::string float_dtype = get_input_tensor(1)->data_type_str();
		INDT_3 << float_dtype << " scale = (" << float_dtype << ") (x_scale[0] / ((" << float_dtype << ")num * y_scale[0]));" << std::endl;
		INDT_3 << "double scaled = ((double)sum) * (double) scale;" << std::endl;
		INDT_3 << "scaled = scaled + (double)y_zero_point[0];" << std::endl;
		INDT_3 << "int t = (int) llround(scaled);" << std::endl;
		INDT_3 << "t = MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "y" << y_idx << " = (" << y->data_type_str() << ") t;" << std::endl;
	}

	void print(std::ostream& dst) const override
	{
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);
	}

	void resolve(void) override
	{
		// Inputs: X, x_scale, x_zero_point, y_scale, y_zero_point
		name_input(0, "x");
		name_scalar_input(1, "x_scale");
		name_scalar_input(2, "x_zero_point");
		name_scalar_input(3, "y_scale");
		name_scalar_input(4, "y_zero_point");

		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();

		// Keep scope small: only dilations==1.
		for (int d : dilations)
			if (d != 1)
				ERROR("Unimplemented: QLinearAveragePool: dilations other than 1");

		const Tensor* y_zero_point = get_input_tensor(4);

		Tensor* rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = y_zero_point->data_type;
		register_output(rv, "y");

		update_pads();
	}
};

} // namespace toC