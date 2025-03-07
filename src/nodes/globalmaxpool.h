namespace toC {

class GlobalMaxPool : public Node {
	public:
	GlobalMaxPool() {
		op_name = "GlobalMaxPool";
	}

	virtual void print(std::ostream &dst) const override
	{
		const Tensor *X=get_input_tensor(0);
		int batch_size = X->data_dim[0];
		int num_channels = X->data_dim[1];

		dst << "\t/* GlobalMaxPool */" << std::endl;
		dst << "\tfor( int32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		dst << "\tfor( int32_t c=0; c<" << num_channels << "; c++ ) {" << std::endl;

		// Initialize max_value to a very small value
		dst << "\t\tfloat max_value = -std::numeric_limits<float>::infinity();" << std::endl;

		std::string in_idx_string = "input[b][c]";  // Start of input element access
		std::string out_idx_string = "output[b][c]"; // Output tensor index

		// Iterate over spatial dimensions
		for( unsigned dim = 2; dim < X->data_dim.size(); dim ++ ) {
			int dim_size = X->data_dim[dim];
			std::string dim_var = "d" + std::to_string(dim-2);
			in_idx_string += "[" + dim_var + "]";
			out_idx_string += "[0]";

			dst << "\t\tfor( int32_t " << dim_var << " = 0; " 
			    << dim_var << " < " << dim_size << "; " 
			    << dim_var << "++ ) {" << std::endl;
		}

		// Update max_value with the max of the current input element
		dst << "\t\t\tmax_value = std::max(max_value, " << in_idx_string << ");" << std::endl;

		// Close loops for spatial dimensions
		for( unsigned dim = 2; dim < X->data_dim.size(); dim ++ ) {
			dst << "\t\t}" << std::endl;
		}

		// Assign the max value to output
		dst << "\t\t" << out_idx_string << " = max_value;" << std::endl;

		// Close loop over batch and channel
		dst << "\t}" << std::endl;
		dst << "\t}" << std::endl;
	}

	virtual void resolve(void) override
	{
		const Tensor *X = get_input_tensor(0);
		name_input(0, "input");
		if(  typeConstraint_plainFloatingPoints(X) == false )
			ERROR("Incorrect input for node"); 

		/* Create output tensors */
		Tensor *rv = new Tensor;
		rv->data_dim.push_back(X->data_dim[0]); // Batch dimension
		rv->data_dim.push_back(X->data_dim[1]); // Channel dimension
		for( unsigned i=2; i<X->data_dim.size(); i++)
			rv->data_dim.push_back(1);  // Reduce spatial dimensions to 1
		rv->data_type = X->data_type;
		register_output(rv, "output");
	}
};

}
