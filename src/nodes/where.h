/* This file is part of onnx2c.
 *
 * Where node.
 * "Return elements, either from X or Y, depending on condition."
 */
namespace toC {

class Where : public Node {
	public:
	Where() {
		op_name = "Where";
	}
	/* Node attributes */

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			ERROR("Unknown attribute for Where: " + a.name());
		}
	}


	virtual void resolve(void) override
	{
		int num_inputs = get_number_of_inputs();
		if(num_inputs != 3)
		{
			ERROR("Number of inputs must be 3");
		}

		const Tensor *condition = get_input_tensor(0);
		const Tensor *x = get_input_tensor(1);
		const Tensor *y = get_input_tensor(2);

		if(condition->data_type != onnx::TensorProto_DataType_BOOL)
		{
			ERROR("The 'condition' tensor must be of type 'bool'");
		}

		if(x->data_type != y->data_type)
		{
			ERROR("'X' and 'Y' tensors must have the same type");
		}

		name_input(0, "condition");
		name_input(1, "X");
		name_input(2, "Y");

		std::vector<int> result_dim=get_input_tensor(0)->data_dim;

		for( int i=1; i<num_inputs; i++ ){
			std::vector<int> tmp;
			multidirectional_broadcast_size(result_dim, get_input_tensor(i)->data_dim, tmp);
			result_dim=tmp;
		}

		Tensor *t = new Tensor;
		t->data_type = x->data_type;
		t->data_dim = result_dim;

		register_output(t, "output");
	}

	virtual void print(std::ostream &dst) const override
	{
		dst << "\t/*" << op_name << "*/" << std::endl;

		// This is essentially copied from 'Elementwise_variadic'
		const Tensor *out = get_output_tensor(0);
		std::string type = out->data_type_str();
		std::vector<std::string> in_idx_strs(get_number_of_inputs());
		std::string out_idx_str;

		// Print out loops over output tensors dimensions
		for( unsigned r=0; r<out->rank(); r++) {
			std::string lv = "i" + std::to_string(r);
			INDT_1 << "for (unsigned " << lv << "=0; ";
			   dst << lv << "<" << out->data_dim[r] << "; ";
			   dst << lv << "++) {" << std::endl;

			// Generate indexing strings to be printed later on.
			// TODO: this is a copy from earlier code. Feels like there might
			// be a more elegant way of doing this.
			for( unsigned i=0; i<get_number_of_inputs(); i++) {
				std::vector<int> pads = get_input_tensor(i)->data_dim;
				std::string idx_str;
				if (pads[r]==1)
					idx_str += "[0]";
				else if(pads[r]!=0)
					idx_str += "[" + lv + "]";

				in_idx_strs[i] += idx_str;
			}
			out_idx_str += "[" + lv + "]";
		}

		INDT_2 << "output" << out_idx_str << " = " << std::endl;
		INDT_3 << "condition" << in_idx_strs[0] << " ? " << "X"
			   << in_idx_strs[1] << " : Y" << in_idx_strs[2] << ";" << std::endl;

		// Close loop over output dimensions
		for( unsigned r=0; r<out->rank(); r++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
}

