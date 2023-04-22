
namespace toC {

class Reshape : public Node {
	public:
	Reshape() {
		op_name = "Reshape";
		data=shape=reshaped=NULL;
		allowzero=0;
	}
	// inputs
	const Tensor *data;
	const Tensor *shape;
	// outputs
	const Tensor *reshaped;

	int32_t allowzero;

	void parseAttributes( onnx::NodeProto &node )
	{
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "allowzero" )
				allowzero = parse_attribute_int(a);
			else
				LOG(ERROR) << "Ignoring attribute " << a.name() << " for node TEMPLATE/" << onnx_name << std::endl;
		}
	}


	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		shape->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		reshaped->print_tensor(dst, !decorate);
	}


	virtual void print(std::ostream &dst) const override
	{
		std::string type = data->data_type_str();

		/* TODO: is there ANY case where a reshape needs to re-order the internal data layout ? */
		/* TODO: and if not - check that at least gcc can get rid of this copy! (So onnx2c doesn't need to) */
		/* TODO: or - can we mark output an onnx2c-alias of input? */
		dst << "\t/*Reshape*/" << std::endl;
		dst << "\t" << type << " *data = (" << type << "*)" << data->cname() << ";" << std::endl;
		dst << "\t" << type << " *reshaped = (" << type << "*)" << reshaped->cname() << ";" << std::endl;

		dst << "\t" << "for( uint32_t i=0; i<" << data->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\treshaped[i] = data[i];" << std::endl;
		dst << std::endl;
	}

	virtual void resolve(void) override
	{
		data = inputs[0];
		shape = inputs[1];

		/* Reshape should allow only int64_t here,
		 * but that is a pointless restriction at this stage and does not play well
		 * with 8-bit quantization.
		 */
		if( typeConstraint_integers(shape) == false )
			ERROR("Incorrect input for node");


		if( shape->initialize == false ) {
			ERROR("Reshaping to a run-time defined shape is not supported");
		}

		if( allowzero != 0) {
			ERROR("Allowzero attribute set. What exactly are you expecting as the output here?");
		}

		std::vector<int> out_data_dim;
		int64_t *new_shape = (int64_t*)(shape->data_buffer);
		bool negative_shape_found=false;
		int negative_shape_at = -1;

		/* Check new shape has no more than 1 negative. Replace zeros */
		uint64_t output_size=1;
		for(unsigned i=0; (int)i<shape->data_num_elem(); i++) {
			int s = new_shape[i];
			if( s < 0 ) {
				if( negative_shape_found )
					ERROR("Bad input: two negatives in reshape's target shape");
				else {
					negative_shape_found = true;
					negative_shape_at=i;
				}
			}
			else if( s == 0 ) {
				if( i >= data->data_dim.size() )
					ERROR("Bad input: Reshape request duplication of input dimension that don't exist");
				s=data->data_dim[i];
			}

			out_data_dim.push_back(s);

			if( s > 0 )
				output_size *= s;
		}

		if( negative_shape_found ) {
			int missing_dim = data->data_num_elem() / output_size;
			// If these don't match, the input is wrong.
			if( output_size * missing_dim != (uint64_t)data->data_num_elem() )
				ERROR("Could not deduce implicit dimension size for Resize node");

			out_data_dim[negative_shape_at] = missing_dim;
		}


		Tensor *rv = new Tensor;
		rv->data_dim = out_data_dim;

		rv->data_type = data->data_type;
		reshaped = rv;
		outputs.push_back(rv);
	}
};
}
