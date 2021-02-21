/* This file is part of onnx2c.
 *
 * Constant node.
 * Outputs a constant tensor.
 * Not quite sure how this is different from an initialized tensor.
 */
namespace toC {

class Constant : public Node {
	public:
	Constant() {
		op_name = "Constant";
		output = NULL;
	}

	const Tensor *output;


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		output->print_tensor(dst, !decorate);
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "value" )
				output = parse_attribute_tensor(a);
			else
				ERROR("Unimplemented parsing of attribute " << a.name());
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* Constant */" << std::endl;
		dst << "\t/* The output is generated as a global tensor */" << std::endl;
		dst << "\t(void)"<<output->cname()<< ";" <<std::endl;
	}


	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( output == NULL )
			ERROR("Constant output tensor should have been resolved by this time");

		Tensor *t = const_cast<toC::Tensor*>(output);
		outputs.push_back(t);
	}
};
}

