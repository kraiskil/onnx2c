/* This file is part of onnx2c.
 *
 * Gather node.
 */
namespace toC {

class Gather : public Node {
	public:
	Gather() {
		op_name = "Gather";
		data=indices=output=NULL;
		axis=0;
	}
	/* Node attributes */
	int axis;

	// input and output
	const Tensor *data;
	const Tensor *indices;
	const Tensor *output;


	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "axis" )
				axis = parse_attribute_int(a);
			else
				ERROR("Unknown attribute for Gather: " + a.name());
		}
	}


	virtual void resolve(void) override
	{
		data = inputs[0];
		indices = inputs[1];

		unsigned a = axis >= 0 ? axis : data->rank()+axis;

		/*
		 * I don't quite understand why the output is calculated like this,
		 * but it passes tests :)
		 */
		Tensor *t = new Tensor;
		unsigned d;
		// output dimension is the same as input, untill 'axis'
		for( d = 0; d<data->rank(); d++) {
			if( d==a )
				break;
			t->data_dim.push_back(data->data_dim[d]);
		}
		// at 'axis', swap over to 'indices' dimensions
		for( auto d: indices->data_dim)
			t->data_dim.push_back(d);
		// and once those are done, collect any remaining input dimension.
		for( d++ ; d<data->rank(); d++)
			t->data_dim.push_back(data->data_dim[d]);

		t->data_type = data->data_type;
		output = t;
		outputs.push_back(t);
	}


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		indices->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		output->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{
		INDT_1 << "/* Gather" << std::endl;
		INDT_1 << "   axis = " << axis << std::endl;
		INDT_1 << " */" << std::endl;

		// The real axis number, counting from 0
		unsigned a = axis >= 0 ? axis : data->rank()+axis;

		// Print out the loops over all output dimensions
		// and at the same time create the indexing strings into the input and output tensors
		// The logic should be the same as above in resolve(void), only here we loop over the
		// output dimensions, not input.
		std::string oidx = output->cname();
		std::string didx = data->cname();
		std::string iidx = indices->cname();
		for( unsigned r=0; r< output->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
			 dst << lv << "<" << output->data_dim[r] << "; ";
			 dst << lv << "++) {" << std::endl;

			oidx += "[" + lv + "]";
			if( r < a )
				didx += "[" + lv + "]";
			else if ( r == a ) {
				didx += "[idx]";
				iidx += "[" + lv  + "]";
			}
			else if ( r <= a+indices->rank()-1 ) {
				iidx += "[" + lv + "]";
			}
			else
				didx += "[" + lv + "]";
		}

		INDT_2 << "int32_t idx = " << iidx << ";" << std::endl;
		INDT_2 << "idx = idx < 0 ? " << data->data_dim[a] << "+idx : idx;" << std::endl;
		INDT_2 << oidx << " = " << didx << ";" << std::endl;

		for( unsigned r=0; r<output->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
}

