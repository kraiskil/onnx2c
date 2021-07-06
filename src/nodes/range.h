/* This file is part of onnx2c.
 *
 * Range node: Generate a tensor containing a sequence 
 * of numbers that begin at start and extends by increments
 * of delta up to limit (exclusive).
 */ 
namespace toC {

class Range : public Node {
	public:
	Range() {
		op_name = "Range";
		start=limit=delta=output=NULL;
	}

	// input and output
	const Tensor *start;
	const Tensor *limit;
	const Tensor *delta;
	const Tensor *output;

	uint32_t output_size;

	template <typename data_type>
	data_type resolve_input_var(const Tensor *t)
	{
		data_type *b;

		b=static_cast<data_type*>(t->data_buffer);
		return b[0];
	}

	template <typename data_type>
	void resolve_limits()
	{
		data_type v_start, v_limit, v_delta;

		v_start = resolve_input_var<data_type>(start);
		v_limit = resolve_input_var<data_type>(limit);
		v_delta = resolve_input_var<data_type>(delta);

		// This line is directly from specification
		output_size = std::max<data_type>( std::ceil( float(v_limit - v_start) / v_delta ) , 0 );
	}

	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{

		if (inputs.size() != 3)
			ERROR("Range node does not have 3 inputs");
		start = inputs[0];
		limit = inputs[1];
		delta = inputs[2];

		if( start->isConst == false )
			ERROR("Unimplemented: non-constant input (start) to Range node");
		if( limit->isConst == false )
			ERROR("Unimplemented: non-constant input (limit) to Range node");
		if( delta->isConst == false )
			ERROR("Unimplemented: non-constant input (delta) to Range node");

		// figure out the contents of start, limit, delta, and the size of output
		if( start->data_type == onnx::TensorProto_DataType_FLOAT )
			resolve_limits<float>();
		else if( start->data_type == onnx::TensorProto_DataType_DOUBLE)
			resolve_limits<double>();
		else if( start->data_type == onnx::TensorProto_DataType_INT32)
			resolve_limits<int32_t>();
		else
			ERROR("Unimplemented data type for Range");


		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor *t = new Tensor;
		t->data_dim.push_back(output_size);
		t->data_type = start->data_type;
		/* Store the created tensor both as reference in this node, and into
		 * the return value vector! */
		output = t;
		outputs.push_back(t);

		/* TODO: optional outputs? */
	}


	/* Print the function parameters - use the order they are introduced in the
	 * ONNX documentation */
	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		start->print_tensor(dst, !decorate);
		dst << ", ";
		limit->print_tensor(dst, !decorate);
		dst << ", ";
		delta->print_tensor(dst, !decorate);
		dst << ", ";
		output->print_tensor(dst, !decorate);
	}


	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{
		std::string dt = start->data_type_str();

		INDT_1 << "/* Range" << std::endl;
		INDT_1 << " */" << std::endl;


		INDT_1 << dt <<" start = " << start->cname() << "[0];" << std::endl;
		INDT_1 << dt <<" delta = " << delta->cname() << "[0];" << std::endl;
		INDT_1 << "for(int i=0; i< "<< output_size << "; ++i) {" << std::endl;
		INDT_2 <<   output->cname() << "[i] = start + (i * delta);" << std::endl;
		INDT_1 << "}" << std::endl;
	}
};
}

