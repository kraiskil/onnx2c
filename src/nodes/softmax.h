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
	Softmax() {
		op_name = "Softmax";
		if( onnx_ir_version < 13 )
			axis = 1;
		else
			axis = -1;
		input=output=NULL;
	}
	// Axis to do the softmax on
	int axis;

	const Tensor *input;
	const Tensor *output;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		input->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		output->print_tensor(dst, !decorate);
	}

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

	void print11(std::ostream &dst) const
	{
		std::string type = input->data_type_str();
		unsigned n_dim = input->data_dim.size();
		std::string expfunc = "expf";
		if( type == "double" )
			expfunc = "exp";
		//TODO fp16?

		unsigned flatten_axis;
		if( axis < 0 ) 
			flatten_axis = input->data_dim.size() + axis;
		else
			flatten_axis = axis;

		dst << "\t/* Softmax 11 (caffe2-style)" << std::endl;
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
		dst << "\t\t" << "max = max>" << input->cname() << idxs << " ? max :" << input->cname() << idxs << ";" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		// Calculate exp() and running sum of innermost flattened dimensions
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		dst << "\t\t" << output->cname() << idxs << " = ";
		dst           << expfunc << "(" << input->cname() << idxs << "-max);" << std::endl;
		dst << "\t\t" << "sum += " << output->cname() << idxs << ";" << std::endl; 
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;
			
		// loop again over inner dimensions to do the division
		for( unsigned i = flatten_axis; i<n_dim; i++) {
			std::string idx = "i" + std::to_string(i);
			dst << "\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << input->data_dim[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}
		dst << "\t\t" << output->cname() <<idxs<<" /= sum;" << std::endl;
		for( unsigned i = flatten_axis; i<n_dim; i++)
			dst << "\t}" << std::endl;

		dst << "\t" << "sum = 0.0;" << std::endl; 
		dst << "\t" << "max = -INFINITY;" << std::endl; 

		
		for( unsigned i = 0; i<flatten_axis; i++ )
			dst << "\t}" << std::endl;
	}

	void print13(std::ostream &dst) const
	{
		std::string type = input->data_type_str();
		unsigned num_dim = input->rank();
		std::string expfunc = "expf";
		if( type == "double" )
			expfunc = "exp";
		//TODO fp16?

		unsigned reduce_axis;
		if( axis < 0 )
			reduce_axis = input->rank() + axis;
		else
			reduce_axis = axis;

		int reduce_axis_size = input->data_dim[reduce_axis];


		INDT_1 << "/* Softmax 13 (TF, pytorch style)" << std::endl;
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
		INDT_3 << "max = max>" << input->cname() << idxs << " ? max :" << input->cname() << idxs << ";" << std::endl;
		INDT_2 << "};" << std::endl;

		// Now loop to calculate sum
		INDT_2 << type << " sum = 0.0;" << std::endl;
		INDT_2 << "for( uint32_t " << ridx << "=0; ";
		   dst <<       ridx << "<" << reduce_axis_size << "; ";
		   dst <<       ridx <<"++ ) {" << std::endl;
		INDT_3 << "sum += " << expfunc << "(" << input->cname() << idxs << " - max);" << std::endl;
		INDT_2 << "};" << std::endl;

		// And last the elementwise softmax
		INDT_2 << "for( uint32_t " << ridx << "=0; ";
		   dst <<       ridx << "<" << reduce_axis_size << "; ";
		   dst <<       ridx <<"++ ) {" << std::endl;
		INDT_3 << output->cname() << idxs << " = ";
		   dst << expfunc << "(" << input->cname() << idxs << " - max)/sum;" << std::endl;
		INDT_2 << "};" << std::endl;

		for( unsigned i = 0; i<num_dim-1; i++ )
			INDT_1 <<"}" << std::endl;
	}


	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to Softmax");

		input = inputs[0];
		if( typeConstraint_allFloatingPoints(input) == false)
			ERROR("Incorrect input for node");

		Tensor *rv = new Tensor;
		rv->data_dim = input->data_dim;
		rv->data_type = input->data_type;
		output = rv;
		outputs.push_back(rv);
	}
};
}
