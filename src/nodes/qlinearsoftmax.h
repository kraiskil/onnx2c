/* This file is part of onnx2c.
 *
 * QLinearSoftmax
 *
 * Quantized softmax (QOperator form).
 *
 * This implementation dequantizes to float, performs a numerically stable
 * softmax along the specified axis, then requantizes to the output type.
 */

#pragma once

#include "node.h"
#include <cmath>

namespace toC {

class QLinearSoftmax : public Node {
	public:
	QLinearSoftmax()
	{
		op_name = "QLinearSoftmax";
		axis = -1;
	}

	int axis;

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		const Tensor* t = get_input_tensor(input_no);
		if (!(t->data_dim.size() == 0 || (t->data_dim.size() == 1 && t->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			if (a.name() == "axis")
				axis = parse_attribute_int(a);
		}
	}

	void resolve(void) override
	{
		// Inputs: X, x_scale, x_zero_point, y_scale, y_zero_point
		name_input(0, "x");
		name_scalar_input(1, "x_scale");
		name_scalar_input(2, "x_zero_point");
		name_scalar_input(3, "y_scale");
		name_scalar_input(4, "y_zero_point");

		const Tensor* x = get_input_tensor(0);
		int ax = axis;
		if (ax < 0)
			ax += (int)x->rank();
		if (ax < 0 || ax >= (int)x->rank())
			ERROR("Invalid axis for QLinearSoftmax");
		if (ax != (int)x->rank() - 1)
			ERROR("Unimplemented: QLinearSoftmax only supports axis == last dimension");

		const Tensor* y_zero_point = get_input_tensor(4);
		Tensor* y = new Tensor;
		y->data_dim = x->data_dim;
		y->data_type = y_zero_point->data_type;
		register_output(y, "y");
	}

	void print(std::ostream& dst) const override
	{
		INDT_1 << "/* QLinearSoftmax */" << std::endl;

		const Tensor* x = get_input_tensor(0);
		const Tensor* y = get_output_tensor(0);

		const int rank = (int)x->rank();
		const uint32_t axis_len = x->data_dim[rank - 1];
		std::string float_dtype = get_input_tensor(1)->data_type_str();
		auto [lower, upper] = y->get_type_bounds();

		// Generate loops for all dims except the last.
		std::string prefix_idx;
		for (int d = 0; d < rank - 1; d++) {
			std::string lv = "i" + std::to_string(d);
			INDT_1 << "for( uint32_t " << lv << "=0; " << lv << "<" << x->data_dim[d] << "; " << lv << "++ )" << std::endl;
			prefix_idx += "[" + lv + "]";
		}

		INDT_1 << "{" << std::endl;
		INDT_2 << float_dtype << " maxv = -INFINITY;" << std::endl;
		INDT_2 << "for( uint32_t j=0; j<" << axis_len << "; j++ ) {" << std::endl;
		INDT_3 << float_dtype << " xf = (" << float_dtype << ")x_scale[0] * ((" << float_dtype << ")((int32_t)x" << prefix_idx
		       << "[j] - (int32_t)x_zero_point[0]));" << std::endl;
		INDT_3 << "if( xf > maxv ) maxv = xf;" << std::endl;
		INDT_2 << "}" << std::endl;

		INDT_2 << "double sum = 0.0;" << std::endl;
		INDT_2 << "for( uint32_t j=0; j<" << axis_len << "; j++ ) {" << std::endl;
		INDT_3 << float_dtype << " xf = (" << float_dtype << ")x_scale[0] * ((" << float_dtype << ")((int32_t)x" << prefix_idx
		       << "[j] - (int32_t)x_zero_point[0]));" << std::endl;
		INDT_3 << "sum += exp((double)(xf - maxv));" << std::endl;
		INDT_2 << "}" << std::endl;

		INDT_2 << "for( uint32_t j=0; j<" << axis_len << "; j++ ) {" << std::endl;
		INDT_3 << float_dtype << " xf = (" << float_dtype << ")x_scale[0] * ((" << float_dtype << ")((int32_t)x" << prefix_idx
		       << "[j] - (int32_t)x_zero_point[0]));" << std::endl;
		INDT_3 << "double p = exp((double)(xf - maxv)) / sum;" << std::endl;
		INDT_3 << "int t = (int) llround(p / (double)y_scale[0] + (double)y_zero_point[0]);" << std::endl;
		INDT_3 << "t = MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "y" << prefix_idx << "[j] = (" << y->data_type_str() << ") t;" << std::endl;
		INDT_2 << "}" << std::endl;

		INDT_1 << "}" << std::endl;
	}
};

} // namespace toC
