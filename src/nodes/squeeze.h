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
	}

	std::vector<int64_t> axes;

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
		const Tensor *data = get_input_tensor(0);
		std::string type = data->data_type_str();

		dst << "\t/*Squeeze*/" << std::endl;
		dst << "\t" << type << " *data = (" << type << "*)input" << ";" << std::endl;
		dst << "\t" << type << " *squeezed= (" << type << "*)output" << ";" << std::endl;

		// TODO: is a memcpy faster?
		dst << "\t" << "for( size_t i=0; i<" << data->data_num_elem() << "; i++ )" << std::endl;
		dst << "\t\t" << "squeezed[i] = data[i];" << std::endl;
		dst << std::endl;
	}
 
	virtual void resolve(void) override
	{
		const Tensor *data = get_input_tensor(0);
		name_input(0, "input");
		if (get_number_of_inputs() == 2) {
			const Tensor *axes_tensor = get_input_tensor(1);
			name_input(1, "axes_tensor");
			if (axes_tensor->isConst == false)
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
		register_output(rv, "output");
	}
};
}
