/* This file is part of onnx2c.
 *
 * Shape node. Return 1d tensor with elements describing the shape of the input.
 */
namespace toC {

class Shape : public Node {
	public:
	Shape()
	{
		op_name = "Shape";
	}

	virtual void resolve(void) override
	{
		const Tensor* data = get_input_tensor(0);
		name_input(0, "data");

		Tensor* t = new Tensor;
		t->data_dim.push_back(data->rank());
		t->data_type = onnx::TensorProto_DataType_INT64;

		// The output of Shape is a compile time constant
		t->isConst = true;
		int64_t* db = new int64_t[data->rank()];
		for (unsigned i = 0; i < data->data_dim.size(); i++)
			db[i] = data->data_dim[i];
		t->data_buffer = (void*)db;

		register_output(t, "output");
	}

	virtual void print(std::ostream& dst) const override
	{
		const Tensor* data = get_input_tensor(0);
		const Tensor* output = get_output_tensor(0);

		// In the odd case of the shape result being a graph output, print it.
		// Othervise those nodes that take Shape output have already
		// read the tensor's data. (I think they all have to, to avoid dynamic size tensors)
		if (output->isIO == false)
			return;

		for (unsigned d = 0; d < data->rank(); d++) {
			INDT_1 << "output[" << d << "]=";
			dst << data->data_dim[d] << ";";
		}
	}
};
} // namespace toC
