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
			if( a.name() == "value" ) {
				LOG(TRACE) << "Adding attribute 'value' as input tensor to node" << std::endl;
				value_tensor = parse_attribute_tensor(a);
				LOG(TRACE) << "\t" << value_tensor->print_trace_dump() << std::endl;
			}
			else
				ERROR("Unimplemented parsing of attribute " << a.name());
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		Tensor *output= get_output_tensor(0);

		dst << "\t/* Constant */" << std::endl;
		dst << "\t/* The output is generated as a global tensor */" << std::endl;

		if( output->isIO == false ) {
			dst << "\t(void)output;" <<std::endl;
			return;
		}
		// most likely this is not what the user wants :)
		LOG(WARNING) << "Constant tensor used as graph output?" << std::endl;

		// Handle the degenerate case (happens in ONNX backend tests for some reason :))
		// where the graph output is a constant.
		if( value_tensor == nullptr )
			ERROR("Constant tensor not resolved");
		std::string dimstr;
		for( unsigned dim=0; dim<value_tensor->rank(); dim++) {
			dimstr += "[d" + std::to_string(dim) + "]";
		}

		print_loops_over_dims(dst, value_tensor, "d", 1);
		INDT_2 << "output" << dimstr << " = " << value_tensor->cname() << dimstr << ";" << std::endl;
		print_loop_closes_over_dims(dst, value_tensor, 1);
	}

	virtual void resolve(void) override
	{
		// value_tensor is the one supplied as the node attribute. It gets
		// copied into the output, as is.
		if( value_tensor == nullptr )
			ERROR("Constant tensor not resolved");
		// "This operator produces a constant tensor."
		value_tensor->isConst = true;
		value_tensor->initialize = true;

		// Just in case someone wants to print out a constant tensor from the graph.
		// Yeah, it's kinda strange... but valid.
		Tensor *rv = new Tensor;
		rv->data_dim = value_tensor->data_dim;
		rv->data_type = value_tensor->data_type;
		rv->isConst = true;
		value_tensor->initialize = true;
		// TODO: remove the above. We have to register the value tensor as output.
		// Otherwise other nodes that access the constant don't see its data.
		register_output(value_tensor, "output");
	}
};
}

