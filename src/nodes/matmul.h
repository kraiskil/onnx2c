/* This file is part of onnx2c.
 *
 * MatMul node.
 * 
 */

#include "node.h"
#include "abstractmatmul.h"

namespace toC {

class MatMul : public AbstractMatMul {
public:
	MatMul() {
		op_name = "MatMul";
	}

	virtual void resolve(void) override;
	void print_multiply_accumulate(std::ostream &dst,
	                               const std::string &y_idx,
	                               const std::string &a_idx,
	                               const std::string &b_idx) const override;
};

void MatMul::resolve(void) {
	Tensor *a = get_input_tensor(0);
	Tensor *b = get_input_tensor(1);

	if (a->data_type != b->data_type) {
		ERROR("Data types of A and B must match in MatMul");
	}

	name_input(0, "A");
	name_input(1, "B");

	Tensor *y = new Tensor;
	y->data_dim = resolve_shape();
	y->data_type = a->data_type;
	register_output(y, "Y");
}

void MatMul::print_multiply_accumulate(std::ostream &dst,
                                       const std::string &y_idx,
                                       const std::string &a_idx,
                                       const std::string &b_idx) const {
	INDT_4 << y_idx << " += " << a_idx << " * " << b_idx << ";" << std::endl;
}

} // namespace

