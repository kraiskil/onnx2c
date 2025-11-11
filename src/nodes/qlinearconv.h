/* This file is part of onnx2c.
 *
 * QLinearConv
 * Quantized Convolution
 */

#include "spatialfilter.h"

namespace toC {
	class QLinearConv : public SpatialFilter {
	public:
		QLinearConv() {
			op_name = "QLinearConv";
		}

		const Tensor* get_X(void) const override { return get_input_tensor(0); }
		const Tensor* get_W(void) const override { return get_input_tensor(3); }

		void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override {
			INDT_3 << "int32_t a = ";
			if (get_number_of_inputs() < 9) {
				dst << "0";
			} else {
				dst << "bias[m]";
			}
			dst << ";" << std::endl;
		}

		void print_output_cell_calc(std::ostream &dst,
									const std::string &x_idx,
									const std::string &w_idx,
									const std::string &y_idx) const override {
			INDT_4 << "a += (x" << x_idx << " - (int32_t)x_zero_point[0]) * (w" << w_idx << " - (int32_t)w_zero_point[0]);" << std::endl;
		}

		void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const override {
			std::string float_dtype = get_input_tensor(1)->data_type_str();
			INDT_3 << float_dtype << " scaled = ((" << float_dtype << ")a) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
			INDT_3 << "scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
			INDT_3 << "y" << y_idx << " = (" << get_output_tensor(0)->data_type_str() << ") roundf(scaled);" << std::endl;
		}

		void print(std::ostream &dst) const override {
			print_header_info_comment(dst);
			print_loop_with_padding_checks(dst);
		}

		void name_scalar_input(unsigned input_no, std::string name) {
			// TODO: Support per-channel quantization
			name_input(input_no, name);
			if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
			      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
				ERROR(name << " must be scalar");
			}
        }

	
		void resolve() override {
			name_input(0, "x");
			name_scalar_input(1, "x_scale");
			name_scalar_input(2, "x_zero_point");

			name_input(3, "w");
			name_scalar_input(4, "w_scale");
			name_scalar_input(5, "w_zero_point");

			name_scalar_input(6, "y_scale");
			name_scalar_input(7, "y_zero_point");

			if (get_number_of_inputs() == 9) {
				name_input(8, "bias");
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
