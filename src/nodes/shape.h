/* This file is part of onnx2c.
 *
 * Shape node. Return 1d tensor with elements describing the shape of the input.
 */ 
namespace toC {

class Shape : public Node {
	public:
	Shape() {
		op_name = "Shape";
		data=output=NULL;
	}

	const Tensor *data;
	const Tensor *output;


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		data = inputs[0];

		Tensor *t = new Tensor;
		t->data_dim.push_back(data->rank());
		t->data_type = onnx::TensorProto_DataType_INT64;
		output = t;
		outputs.push_back(t);
	}


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		output->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Shape */" << std::endl;
		for( unsigned d = 0; d<data->rank(); d++ ) {
			INDT_1 << output->cname() << "["<<d<<"]=";
			dst    << data->data_dim[d] << ";";
		}
	}
};
}

