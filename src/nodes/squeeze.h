/* This file is part of onnx2c.
 *
 * Squeeze.
 * "Remove single-dimensional entries from the shape of a tensor."
 * I.e. flatten dimensions of size 1.
 */
namespace toC {

class Squeeze : public Node {
	public:
	Squeeze() {
		op_name = "Squeeze";
		data=squeezed=NULL;
	}
	// inputs
	const Tensor *data;
	const Tensor *axes_tensor;
	// outputs
	const Tensor *squeezed;

	std::vector<int64_t> axes;

	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		data->print_tensor_as_const(dst, !decorate);
		dst << ", ";
		squeezed->print_tensor(dst, !decorate);
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override
	{
		for( const auto& a : node.attribute() ) {
			if( a.name() == "axes" )
				axes = parse_attribute_ints(a);
			else
				ERROR("Bad attribute " << a.name() << " to squeeze");
		}
	}

	virtual void print(std::ostream &dst) const override
	{
		std::string type = data->data_type_str();

		dst << "\t/*Squeeze*/" << std::endl;
		dst << "\t" << type << " *data = (" << type << "*)" << data->cname() << ";" << std::endl;
		dst << "\t" << type << " *squeezed= (" << type << "*)" << squeezed->cname() << ";" << std::endl;

		// TODO: is a memcpy faster?
		dst << "\t" << "for( uint32_t i=0; i<" << data->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\t" << "squeezed[i] = data[i];" << std::endl;
		dst << std::endl;
	}
 
	virtual void resolve(void) override
	{
		data = inputs[0];
		if (inputs.size() == 2) {
			axes_tensor = inputs[1];
			if (axes_tensor->initialize == false)
				ERROR("provided axes are dynamic, not implmeneted");
			for( unsigned i=0; (int)i<axes_tensor->data_num_elem(); i++) {
				int64_t *rd = (int64_t*)axes_tensor->data_buffer;  // axes data must be int64
				axes.push_back(rd[i]);
			}
		}

		// if not given, all dimensions with size 1 are squeezed
		if( axes.size() == 0 )
			for( unsigned i=0; i<data->data_dim.size(); i++ )
				if( data->data_dim[i] == 1 )
					axes.push_back(i);


		if( axes.size() == 0 )
			ERROR("No axes to squeeze away?");

		// negative axes means counted from "end"
		for( auto &a : axes )
			if( a<0 )
				a = data->data_dim.size() + a;

		// Push to the resultant tensor only dimension not squeezed away
		Tensor *rv = new Tensor;
		for( unsigned i=0; i<data->data_dim.size(); i++ ) {
			bool to_be_squeezed=false;
			for( int a : axes )
				if( a == (int)i )
					to_be_squeezed=true;

			if( to_be_squeezed ){
				if( data->data_dim[i] != 1 )
					ERROR("Attempting to squeeze an unsqeezable dimension");
			}
			else
				rv->data_dim.push_back(data->data_dim[i]);
		}

		rv->data_type = data->data_type;
		squeezed = rv;
		outputs.push_back(rv);
	}
};
}
