/* This file is part of onnx2c.
 *
 * Generic node for variadic input Operations with one output.
 * Calculates elementvise out[idx] = <op>(in1[idx],...)
 * Operation is defined as attribute passed to the constructor.
 * - max, mean, min, sum
 */
namespace toC {

class Elementwise_variadic : public Node {
	public:

	// Each instance of this class should override this lambda with the operation of the node type.
	// Inputs: the stream to print to and indexing arrays for each of the input vectors (padded for broadcasting)
	// See the implementations below for clarification :)
	std::function<void (std::ostream &, const std::vector<std::string> &)> operation =
		[](std::ostream &a, const std::vector<std::string> &b){ ERROR("onnx2c internal error"); };

	std::vector<const Tensor *>in;
	const Tensor *out;


	Elementwise_variadic(std::string op) {
		op_name = op;
		out=NULL;

		if( op == "Min" )
			operation = [this](std::ostream &dst, const std::vector<std::string> &idxs)
				{
					INDT_3 << "MIN(" << in[0]->cname() << idxs[0] << ", " << std::endl;
					for(unsigned i=0; i<in.size()-1; i++)
						INDT_3 << "MIN(" << in[i]->cname() << idxs[i] << ", " << std::endl;
					INDT_4 << in[in.size()-1]->cname() << idxs[in.size()-1];
					for(unsigned i=0; i<in.size(); i++)
						dst << ")";
					dst << ";" << std::endl;
				};
		else if( op == "Mean" )
			operation = [this](std::ostream &dst, const std::vector<std::string> &idxs)
				{
					INDT_3 << "("  << in[0]->cname() << idxs[0] << std::endl;
					for(unsigned i=1; i<in.size(); i++)
						INDT_3 << " + " << in[i]->cname() << idxs[i] << std::endl;
					INDT_3 << ")/" << in.size() << ";" << std::endl;
				};
		else if( op == "Max" )
			operation = [this](std::ostream &dst, const std::vector<std::string> &idxs)
				{
					INDT_3 << "MAX(" << in[0]->cname() << idxs[0] << ", " << std::endl;
					for(unsigned i=0; i<in.size()-1; i++)
						INDT_3 << "MAX(" << in[i]->cname() << idxs[i] << ", " << std::endl;
					INDT_4 << in[in.size()-1]->cname() << idxs[in.size()-1];
					for(unsigned i=0; i<in.size(); i++)
						dst << ")";
					dst << ";" << std::endl;
				};
		else if (op == "Sum" )
			operation = [this](std::ostream &dst, const std::vector<std::string> &idxs)
				{
					INDT_3 << "("  << in[0]->cname() << idxs[0] << std::endl;
					for(unsigned i=1; i<in.size(); i++)
						INDT_3 << " + " << in[i]->cname() << idxs[i] << std::endl;
					INDT_3 << ");" << std::endl;
				};
		else
			ERROR("Elementwise_variadic: operand " + op + " not implemented");
	}

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		for( auto i : in ) {
			i->print_tensor_as_const(dst, !decorate);
			dst << ", ";
		}
		out->print_tensor(dst, !decorate);
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			ERROR("unknown attribute");
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		std::string type = out->data_type_str();
		std::vector<std::string> in_idx_strs(in.size());
		std::string out_idx_str;
		INDT_1 << "/* " << op_name  << std::endl;
		INDT_1 << "   Implemented with Elementwise_variadic template." << std::endl;
		INDT_1 << " */" << std::endl;


		// Print out loops over output tensors dimensions
		for( unsigned r=0; r<out->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
			   dst << lv << "<" << out->data_dim[r] << "; ";
			   dst << lv << "++) {" << std::endl;

			// Generate indexing strings to be printed later on.
			// TODO: this is a copy from earlier code. Feels like there might
			// be a more elegant way of doing this.
			for( unsigned i=0; i<in.size(); i++) {
				std::vector<int> pads = in[i]->data_dim;
				std::string idx_str;
				if (pads[r]==1)
					idx_str += "[0]";
				else if(pads[r]!=0)
					idx_str += "[" + lv + "]";

				in_idx_strs[i] += idx_str;
			}
			out_idx_str += "[" + lv + "]";
		}

		// apply operation over input tensors, for each output element separately.
		INDT_2 << out->cname() << out_idx_str << " = " << std::endl;
		operation( dst, in_idx_strs);

		// Close loop over output dimensions
		for( unsigned r=0; r<out->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		// There can be 1-N inputs.
		int num_inputs = inputs.size();

		std::vector<int> result_dim=inputs[0]->data_dim;
		in.push_back(inputs[0]);
		for( int i=1; i<num_inputs; i++ ){
			std::vector<int> tmp;
			multidirectional_broadcast_size(result_dim, inputs[i]->data_dim, tmp);
			result_dim=tmp;
			in.push_back(inputs[i]);
		}

		Tensor *t = new Tensor;
		t->data_dim = result_dim;
		t->data_type = in[0]->data_type;
		out = t;
		outputs.push_back(t);
	}
};
}

