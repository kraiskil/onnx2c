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

	std::vector<int> a_dim = a->data_dim;
	if (a_dim.size() == 1) {
		a_dim.insert(a_dim.begin(), 1);
	} else if (a_dim.size() == 0) {
		ERROR("Scalar input not supported for MatMul");
	}

	std::vector<int> b_dim = b->data_dim;
	if (b_dim.size() == 1) {
		b_dim.push_back(1);
	} else if (a_dim.size() == 0) {
		ERROR("Scalar input not supported for MatMul");
	}

	assert(a_dim.size() >= 2);
	assert(b_dim.size() >= 2);

	for (unsigned i = 2; i < a_dim.size() && i < b_dim.size(); i++) {
		if (a_dim[a_dim.size() - i - 1] != b_dim[b_dim.size() - i - 1]) {
			ERROR("Invalid broadcast dimensions for MatMul");
		}
	}

	std::vector<int> y_dim;

	if (a_dim.size() > b_dim.size()) {
		y_dim.insert(y_dim.end(), a_dim.begin(), a_dim.end() - 2);
	} else {
		y_dim.insert(y_dim.end(), b_dim.begin(), b_dim.end() - 2);
	}
	
	y_dim.push_back(a_dim[a_dim.size() - 2]);
	y_dim.push_back(b_dim[b_dim.size() - 1]);

	Tensor *y = new Tensor;
	y->data_dim = y_dim;
	y->data_type = a->data_type;
	register_output(y, "Y");
}

static std::string matmul_broadcast(std::string name, const Tensor *t, int to_rank) {
	assert(t->rank() >= 2);

	std::ostringstream dst;
	dst << name;
	for (int i = 0; i < (int)t->data_dim.size() - 2; i++) {
		dst << "[i" << (to_rank - t->data_dim.size() + i) << "]";
	}
	return dst.str();
}

void MatMul::print(std::ostream &dst) const {
	INDT_1 << "/* MatMul */" << std::endl;

	Tensor *a = get_input_tensor(0);
	Tensor *b = get_input_tensor(1);
	Tensor *y = get_output_tensor(0);

	for (int i = 0; i < (int)y->rank() - 2; i++) {
		std::string lv = "i" + std::to_string(i);
		INDT_1 << "for (unsigned " << lv << "=0; " << lv << "<" << y->data_dim[i] << "; " << lv << "++)" << std::endl;
	}

	std::string a_idx = matmul_broadcast("A", a, y->rank());
	if (a->data_dim.size() == 1) {
		a_idx += "[k]";
	} else {
		a_idx += "[i][k]";
	}

	std::string b_idx = matmul_broadcast("B", b, y->rank());
	if (b->data_dim.size() == 1) {
		b_idx += "[k]";
	} else {
		b_idx += "[k][j]";
	}

	std::string y_idx = matmul_broadcast("Y", y, y->rank()) + "[i][j]";

	INDT_1 << "{" << std::endl;
	
	INDT_2 << "for (unsigned i = 0; i < " << y->data_dim[y->rank() - 2] << "; i++)" << std::endl;
	INDT_2 << "for (unsigned j = 0; j < " << y->data_dim[y->rank() - 1] << "; j++)" << std::endl;
	INDT_2 << "{" << std::endl;
	INDT_2 << y_idx << " = 0;" << std::endl;
	INDT_3 << "for (unsigned k = 0; k < " << a->data_dim[a->rank() - 1] << "; k++)" << std::endl;
	INDT_4 << y_idx << " += " << a_idx << " * " << b_idx << ";" << std::endl;
	INDT_2 << "}" << std::endl;

	INDT_1 << "}" << std::endl;
}

} // namespace

