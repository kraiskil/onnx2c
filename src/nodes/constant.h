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
	}

	Tensor *value_tensor = nullptr;

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "value" )
				value_tensor = parse_attribute_tensor(a);
			else
				ERROR("Unimplemented parsing of attribute " << a.name());
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/* Constant */" << std::endl;
		dst << "\t/* The output is generated as a global tensor */" << std::endl;
		dst << "\t(void)output;" <<std::endl;
	}

	virtual void resolve(void) override
	{
		if( value_tensor == nullptr )
			ERROR("Constant tensor not resolved");
		// "This operator produces a constant tensor."
		value_tensor->isConst = true;
		register_output(value_tensor, "output");
	}
};
}

