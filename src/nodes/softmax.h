/* This file is part of onnx2c.
 *
 * Softmax.

 * Version 11 and earlier:
 * Calculates
 * exp( x - max ) / sum( exp( x - max ) )
 * sum() is calculated over batches consisting of all
 * dimensions 'deeper than'/'more inner than'/'right of'
 * given attribute 'axis'.
 * (NB: this might be a bad interpretation of the specification
 *  but backend tests pass :))
 *
 * Version 13:
 * Specs say:
 * Softmax(input, axis) = Exp(input) / ReduceSum(Exp(input), axis=axis, keepdims=1)
 * (this seems to be in deprecated Tensorflow syntax, according to quick searching)
 * Seems to be "standard" behaviour: https://github.com/onnx/onnx/issues/2289
 * I.e. for a tensor of shape (X,Y,Z) the element [x, y, z] is divided by the sum of
 * elements [x [1..Y] z].
 * The trick with (x-max) is to accomodate big values of x (where exp(x)->inf).
 * subtracting max doesn't change the result.
 */
namespace toC {

class Softmax : public Node {
	public:
	Softmax(std::string op) {
		op_name = op;
		
		if (op == "LogSoftmax") {
			is_log_softmax = true;
		} else {
			assert(op == "Softmax");
			is_log_softmax = false;
		}
		
		if( onnx_ir_version < 13 )
			axis = 1;
		else
			axis = -1;
	}

	bool is_log_softmax;

	// Axis to do the softmax on
	int axis;

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto& a : node.attribute() ) {
			if( a.name() == "axis" ) {
				if( a.type() != onnx::AttributeProto_AttributeType_INT )
					ERROR("Bad attribute " << a.name());
				if( a.has_i() == false )
					ERROR("Bad attribute " << a.name());
				axis = a.i();
			}
			else
				ERROR("Unknown attribute " << a.name());
		}
	}

	void print(std::ostream &dst) const override
	{
		if( onnx_ir_version > 12 )
			print13(dst);
		else
			print11(dst);
	}

	std::string expfunc() const {
		switch (get_input_tensor(0)->data_type) {
			case onnx::TensorProto_DataType_FLOAT:
				return "expf";
			case onnx::TensorProto_DataType_DOUBLE:
				return "exp";
			default:
				ERROR("exp function is not available for type " << get_input_tensor(0)->data_type_str());
		}
	}

	std::string logfunc() const {
		switch (get_input_tensor(0)->data_type) {
			case onnx::TensorProto_DataType_FLOAT:
				return "logf";
			case onnx::TensorProto_DataType_DOUBLE:
				return "log";
			default:
				ERROR("log function is not available for type " << get_input_tensor(0)->data_type_str());
		}
	}

	void print11(std::ostream &dst) const
	{
		const Tensor *input=get_input_tensor(0);
		std::string type = input->data_type_str();
		unsigned n_dim = input->data_dim.size();

		unsigned flatten_axis;
		if( axis < 0 ) 
			flatten_axis = input->data_dim.size() + axis;
		else
			flatten_axis = axis;

		dst << "\t/* " << op_name << " 11 (caffe2-style)" << std::endl;
		dst << "\t * Implemented with Softmax template." << std::endl;
		dst << "\t * axis = " << axis << std::endl;
		dst << "\t */" << std::endl; 
		dst << "\t" << type << " sum = 0.0;" << std::endl;
		dst << "\t" << type << " max = -INFINITY;" << std::endl;
		dst << std::endl;

		std::string idxs;
		for( unsigned i = 0; i<n_dim; i++)
			idxs += "[i" + std::to_string(i) + "]";

		for( unsigned i = 0; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// Calculate max of the flattened inner dimensions, and close those loops
		dst << "\t\t" << "max = max>input" << idxs << " ? max : input" << idxs << ";" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		// Calculate exp() and running sum of innermost flattened dimensions
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		INDT_2 << "output"<< idxs << " = ";
		dst           << expfunc() << "(input" << idxs << "-max);" << std::endl;
		INDT_2 << "sum += output" << idxs << ";" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;
			
		// loop again over inner dimensions to do the division
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		INDT_2 << "output" << idxs <<" /= sum;" << std::endl;
		if (is_log_softmax) {
			INDT_2 << "output" << idxs <<" = " << logfunc() << "(output" << idxs << ");" << std::endl;
		}

		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		dst << "\t" << "sum = 0.0;" << std::endl; 
		dst << "\t" << "max = -INFINITY;" << std::endl; 

		
		for( unsigned i = 0; i<flatten_axis; i++ )
			dst << "\t}" << std::endl;
	}

	void print13(std::ostream &dst) const
	{
		const Tensor *input=get_input_tensor(0);

		std::string type = input->data_type_str();
		unsigned num_dim = input->rank();

		unsigned reduce_axis;
		if( axis < 0 )
			reduce_axis = input->rank() + axis;
		else
			reduce_axis = axis;

		int reduce_axis_size = input->data_dim[reduce_axis];


		INDT_1 << "/* Softmax 13 (TF, pytorch style)" << std::endl;
		INDT_1 << " * Implemented with Softmax template." << std::endl;
		INDT_1 << " * axis = " << axis << std::endl;
		INDT_1 << " */" << std::endl;

		// Loop over all tensor elements, leaving the axis along which to calculate
		// softmax as the innermost loop.
		std::string idxs;
		for( unsigned i = 0; i<num_dim; i++) {
			idxs += "[i" + std::to_string(i) + "]";
			if (i==reduce_axis)
				continue;
			std::string idx = "i" + std::to_string(i);
			INDT_1 << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// Loop over the reduction axis three times, first calculate the max, then sum, then the elements
		std::string ridx = "i" + std::to_string(reduce_axis);
		INDT_2 << type << " max = -INFINITY;" << std::endl;
		INDT_2 << "for( uint32_t " << ridx << "=0; ";
		   dst <<       ridx << "<" << reduce_axis_size << "; ";
		   dst <<       ridx <<"++ ) {" << std::endl;
		INDT_3 << "max = max>input" << idxs << " ? max :input" << idxs << ";" << std::endl;
		INDT_2 << "};" << std::endl;

		// Now loop to calculate sum
		INDT_2 << type << " sum = 0.0;" << std::endl;
		INDT_2 << "for( uint32_t " << ridx << "=0; ";
		   dst <<       ridx << "<" << reduce_axis_size << "; ";
		   dst <<       ridx <<"++ ) {" << std::endl;
		INDT_3 << "sum += " << expfunc() << "(input" << idxs << " - max);" << std::endl;
		INDT_2 << "};" << std::endl;

		// And last the elementwise softmax
		INDT_2 << "for( uint32_t " << ridx << "=0; ";
		   dst <<       ridx << "<" << reduce_axis_size << "; ";
		   dst <<       ridx <<"++ ) {" << std::endl;
		INDT_3 << "output" << idxs << " = ";
		if (is_log_softmax) dst << logfunc() << "(";
		   dst << expfunc() << "(input" << idxs << " - max)/sum";
		if (is_log_softmax) dst << ")";
		   dst << ";" << std::endl;
		INDT_2 << "};" << std::endl;

		for( unsigned i = 0; i<num_dim-1; i++ )
			INDT_1 <<"}" << std::endl;
	}


	virtual void resolve(void) override
	{
		if( get_number_of_inputs() != 1 )
			ERROR("wrong number of inputs to Softmax");

		name_input(0, "input");

		Tensor *rv = new Tensor;
		rv->data_dim = get_input_tensor(0)->data_dim;
		rv->data_type = get_input_tensor(0)->data_type;
		register_output(rv, "output");
	}
};
}
