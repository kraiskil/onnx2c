/* This file is part of onnx2c.
 *
 * MatMul node.
 * 
 */

#include "node.h"

namespace toC {

class MatMul : public Node {
	public:
	MatMul() {
		op_name = "MatMul";
	}

	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};

void MatMul::resolve(void) {
	Tensor *a = get_input_tensor(0);
	Tensor *b = get_input_tensor(1);

	assert(a->data_type == b->data_type);

	name_input(0, "A");
	name_input(1, "B");

	assert(a->rank() >= 1);
	assert(b->rank() >= 1);

	std::vector<int> y_dim;

	if (a->rank() > 2 || b->rank() > 2) {
		for (unsigned i = 2; i < a->rank() && i < b->rank(); i++) {
			if (a->data_dim[a->rank() - i - 1] != b->data_dim[b->rank() - i - 1]) {
				ERROR("Invalid broadcast dimensions for MatMul");
			}
		}

		int k_dim_a = a->data_dim[a->rank() - 1];
		int k_dim_b = b->rank() > 1 ? b->data_dim[b->rank() - 2] : b->data_dim[0];
		if (k_dim_a != k_dim_b) {
			ERROR("Reduction dimension mismatch in MatMul");
		}

		if (a->data_dim.size() > b->data_dim.size()) {
			y_dim.insert(y_dim.end(), a->data_dim.begin(), a->data_dim.end() - 2);
		} else {
			y_dim.insert(y_dim.end(), b->data_dim.begin(), b->data_dim.end() - 2);
		}
	}
	
	if (a->rank() > 1) {
		y_dim.push_back(a->data_dim[a->rank() - 2]);
	}
	
	if (b->rank() > 1) {
		y_dim.push_back(b->data_dim[b->rank() - 1]);
	}

	Tensor *y = new Tensor;
	y->data_dim = y_dim;
	y->data_type = a->data_type;
	register_output(y, "Y");
}

void MatMul::print(std::ostream &dst) const {
	INDT_1 << "/* MatMul */" << std::endl;

	Tensor *a = get_input_tensor(0);
	Tensor *b = get_input_tensor(1);
	Tensor *y = get_output_tensor(0);

	// Number of dimensions to broadcast over
	int broadcast_dims = y->rank();
	if (a->rank() > 1) {
		broadcast_dims--;
	}
	if (b->rank() > 1) {
		broadcast_dims--;
	}

	for (int i = 0; i < broadcast_dims; i++) {
		std::string lv = "i" + std::to_string(i);
		INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << y->data_dim[i] << "; " << lv << "++)" << std::endl;
	}

	std::string a_idx = "A";
	if (a->rank() == 1) {
		a_idx += "[k]";
	} else {
		for (int i = 0; i < (int)a->rank() - 2; i++) {
			a_idx += "[i" + std::to_string(broadcast_dims - ((int)a->rank() - 2) + i) + "]";
		}
		a_idx += "[i][k]";
	}

	std::string b_idx = "B";
	if (b->rank() == 1) {
		b_idx += "[k]";
	} else {
		for (int i = 0; i < (int)b->rank() - 2; i++) {
			b_idx += "[i" + std::to_string(broadcast_dims - ((int)b->rank() - 2) + i) + "]";
		}
		b_idx += "[k][j]";
	}

	std::string y_idx;
	if (y->is_scalar()) {
		y_idx = "*Y";
	} else {
		y_idx = "Y";

		for (int i = 0; i < broadcast_dims; i++) {
			y_idx += "[i" + std::to_string(i) + "]";
		}

		if (a->rank() > 1) {
			y_idx += "[i]";
		}

		if (b->rank() > 1) {
			y_idx += "[j]";
		}
	}

	int i_dim = (a->rank() > 1) ? a->data_dim[a->rank() - 2] : 1;
	int j_dim = (b->rank() > 1) ? b->data_dim[b->rank() - 1] : 1;
	int k_dim = a->data_dim[a->rank() - 1];

	INDT_1 << "{" << std::endl;
	
	INDT_2 << "for (unsigned i = 0; i < " << i_dim << "; i++)" << std::endl;
	INDT_2 << "for (unsigned j = 0; j < " << j_dim << "; j++)" << std::endl;
	INDT_2 << "{" << std::endl;
	INDT_3 << y_idx << " = 0;" << std::endl;
	INDT_3 << "for (unsigned k = 0; k < " << k_dim << "; k++)" << std::endl;
	INDT_4 << y_idx << " += " << a_idx << " * " << b_idx << ";" << std::endl;
	INDT_2 << "}" << std::endl;

	INDT_1 << "}" << std::endl;
}

} // namespace

