/* This file is part of onnx2c.
 *
 * Identity node.
 *
 */

#include "node.h"

namespace toC {

class Identity : public Node {
	public:
	Identity()
	{
		op_name = "Identity";
	}

	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

void Identity::resolve(void)
{
	Tensor* input = get_input_tensor(0);
	name_input(0, "input");

	Tensor* t = new Tensor;
	t->data_dim = input->data_dim;
	t->data_type = input->data_type;
	register_output(t, "output");
}

void Identity::print(std::ostream& dst) const
{
	LOG(INFO) << "Emitting unnecessary code for Identity node " << onnx_name << "." << std::endl;

	INDT_1 << "/* Identity */" << std::endl;

	Tensor* input = get_input_tensor(0);
	std::string type = input->data_type_str();

	INDT_1 << "const " << type << "* input_ptr = (const " << type << "*) input;" << std::endl;
	INDT_1 << type << "* output_ptr = (" << type << "*) output;" << std::endl;

	INDT_1 << "for (unsigned i = 0; i < " << input->data_num_elem() << "; i++)" << std::endl;
	INDT_2 << "output_ptr[i] = input_ptr[i];" << std::endl;
}

} // namespace toC
