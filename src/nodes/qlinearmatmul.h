/* This file is part of onnx2c.
 *
 * QLinearMatMul
 * 
 */

#include "abstractmatmul.h"

namespace toC {
    class QLinearMatMul : public AbstractMatMul {
    public:
        QLinearMatMul() {
            op_name = "QLinearMatMul";
        }

        Tensor* get_a() const override { return get_input_tensor(0); }
        Tensor* get_b() const override { return get_input_tensor(3); }

        void print_initialize(std::ostream &dst, const std::string &y_idx) const override {
            INDT_3 << "int32_t x " << " = 0;" << std::endl;
        }

        void print_multiply_accumulate(std::ostream &dst,
                                       const std::string &y_idx,
                                       const std::string &a_idx,
                                       const std::string &b_idx) const override {

            INDT_4 << "x += ((int32_t)" << a_idx << " - a_zero_point[0]) * "
                   << "((int32_t)" << b_idx << " - b_zero_point[0]);" << std::endl;
        }

        void print_finalize(std::ostream& dst, const std::string& y_idx) const override {
            std::string float_dtype = get_input_tensor(1)->data_type_str();
            INDT_3 << float_dtype << " scaled = ((" << float_dtype << ")x) * (a_scale[0] * b_scale[0]) / y_scale[0];" << std::endl;
            INDT_3 << "scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
            INDT_3 << y_idx << " = (" << get_output_tensor(0)->data_type_str() << ") froundf(scaled);" << std::endl;
        }

        void name_scalar_input(unsigned input_no, std::string name) {
            // TODO: Support per-channel quantization
            name_input(input_no, name);
            if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
			      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
				ERROR(name << " must be scalar");
			}
        }

        void resolve(void) override {
            name_input(0, "A");
            name_scalar_input(1, "a_scale");
            name_scalar_input(2, "a_zero_point");

            name_input(3, "B");
            name_scalar_input(4, "b_scale");
            name_scalar_input(5, "b_zero_point");

            name_input(6, "y_scale");
            name_scalar_input(7, "y_zero_point");

            Tensor *y_zero_point = get_input_tensor(7);

            Tensor *y = new Tensor;
            y->data_dim = resolve_shape();
            y->data_type = y_zero_point->data_type;
            register_output(y, "Y");
        }
    };
}