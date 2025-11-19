/* This file is part of onnx2c.
 *
 * Expand node.
 */
#include "expand.h"
#include <deque>
namespace toC {

/*
 * The output size calculation algorithm is:
 * -create a tensor with dimensions given in 'shape' input
 * -pad this tensor or the input with dummy dimensions "to the left", i.e. outermost dimensions so ranks agree
 * -for each dimension, padded shape and input should have same size, or that dimension size of one must be 1
 * -output dimensions are the maximum of padded shape and input size
 */
std::vector<int32_t> Expand::resolve_output_shape(void) const
{
	std::deque<int32_t> unpadded_input_sizes;
	for (auto d : input->data_dim)
		unpadded_input_sizes.push_back(d);

	int64_t* shape_contents = static_cast<int64_t*>(shape->data_buffer);
	std::deque<int32_t> shape_data;
	for (int i = 0; i < shape->data_num_elem(); i++)
		shape_data.push_back(shape_contents[i]);

	unsigned output_dims = std::max((unsigned)shape_data.size(), input->rank());

	// broadcasting is "right alinged", so pad size vectors at the start
	while (unpadded_input_sizes.size() < output_dims)
		unpadded_input_sizes.push_front(1);
	while (shape_data.size() < output_dims)
		shape_data.push_front(1);

	std::vector<int32_t> output_shape;
	for (unsigned i = 0; i < output_dims; i++) {
		// TODO: do we trust the input is well formatted, and these two are equal, or one of them is 1?
		int32_t outs = std::max(shape_data[i], unpadded_input_sizes[i]);
		output_shape.push_back(outs);
	}
	return output_shape;
}

/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void Expand::resolve(void)
{
	input = get_input_tensor(0);
	name_input(0, "input");
	shape = get_input_tensor(1);
	name_input(1, "shape");

	if (shape->isConst == false)
		ERROR("Unimplemented: Expand operand with non-constant input (shape).");

	std::vector<int32_t> output_shape;
	output_shape = resolve_output_shape();

	Tensor* t = new Tensor;
	t->data_dim = output_shape;
	t->data_type = input->data_type;
	output = t;
	register_output(t, "output");
}

/* Body of the node implementing function */
void Expand::print(std::ostream& dst) const
{
	INDT_1 << "/* Expand */" << std::endl;

	// loop over output
	std::string i_idxs;
	std::string o_idxs;
	for (unsigned i = 0; i < output->rank(); i++) {
		std::string i_str = std::to_string(i);
		std::string o_idx = "o" + i_str + "";
		o_idxs += "[" + o_idx + "]";
		std::string i_idx;
		unsigned input_pads = output->rank() - input->rank();
		if (input_pads > i)
			;
		else if (input->data_dim[i - input_pads] == 1)
			i_idxs += "[0]";
		else
			i_idxs += "[o" + i_str + "]";
		INDT_1 << "for( uint32_t " << o_idx << "=0; ";
		dst << o_idx << "<" << output->data_dim[i] << "; ";
		dst << o_idx << "++) {" << std::endl;
	}

	INDT_2 << "output" << o_idxs << " = input" << i_idxs << ";" << std::endl;

	for (unsigned i = 0; i < output->rank(); i++)
		INDT_1 << "}" << std::endl;
}

} // namespace toC
