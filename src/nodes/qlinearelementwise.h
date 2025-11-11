/* This file is part of onnx2c.
 *
 * QLinearElementwise node.
 * 
 */

#include "node.h"

namespace toC {
	class QLinearElementwise : public Node {
	public:
		QLinearElementwise(std::string op) {
			op_name = op;
		}

		// QLinearElementwise Operators do not support per-channel quantization
		void name_scalar_input(unsigned input_no, std::string name) {
			name_input(input_no, name);
			if (!(get_input_tensor(input_no)->data_dim.size() == 0 ||
			      (get_input_tensor(input_no)->data_dim.size() == 1 && get_input_tensor(input_no)->data_dim[0] == 1))) {
				ERROR(name << " must be scalar");
			}
        }

		void resolve() override {
			name_input(0, "A");
			name_scalar_input(1, "A_scale");
			name_scalar_input(2, "A_zero_point");

			name_input(3, "B");
			name_scalar_input(4, "B_scale");
			name_scalar_input(5, "B_zero_point");

			name_scalar_input(6, "C_scale");
			name_scalar_input(7, "C_zero_point");

			Tensor* a = get_input_tensor(0);
			Tensor* b = get_input_tensor(3);

			std::vector<int> result_dim;
			multidirectional_broadcast_size(a->data_dim, b->data_dim, result_dim);

			Tensor *c = new Tensor;
			c->data_dim = result_dim;
			c->data_type = a->data_type;
			register_output(c, "C");
		}
		
		void print(std::ostream &dst) const override {
			INDT_1 << "/* " << op_name << " (QLinearElementwise) */" << std::endl;

			Tensor* a = get_input_tensor(0);
			Tensor* b = get_input_tensor(3);
			Tensor* c = get_output_tensor(0);

			for (unsigned r = 0; r < c->rank(); r++) {
				std::string lv = "i" + std::to_string(r);
				INDT_1 << "for (size_t " << lv << " = 0; " << lv << " < " << c->data_dim[r] << "; " << lv << "++)" << std::endl;
			}

			INDT_1 << "{" << std::endl;

			std::string a_idx = broadcast(a, "A", c->rank());
			std::string b_idx = broadcast(b, "B", c->rank());
			std::string c_idx = broadcast(c, "C", c->rank());

			std::string float_type = get_input_tensor(1)->data_type_str();

			INDT_2 << float_type << " a = ((" << float_type << ")" << a_idx << " - A_zero_point[0]) * A_scale[0];" << std::endl;
			INDT_2 << float_type << " b = ((" << float_type << ")" << b_idx << " - B_zero_point[0]) * B_scale[0];" << std::endl;

			INDT_2 << float_type << " c = ";
			if (op_name == "QLinearAdd") {
				dst << "a + b";
			} else if (op_name == "QLinearMul") {
				dst << "a * b";
			} else {
				ERROR("Unsupported QLinearElementwise operation: " << op_name);
			}
			dst << ";" << std::endl;

			INDT_2 << c_idx << " = roundf(c / C_scale[0] + C_zero_point[0]);" << std::endl;

			INDT_1 << "}" << std::endl;
		}
	};

} // namespace

