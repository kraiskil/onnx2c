/* This file is part of onnx2c.
 *
 * Pad node.
 */
#include "pad.h"
namespace toC {


/* Parse attributes, if this node has them. */
void Pad::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "mode" )
			mode = parse_attribute_string(a);
		else if( a.name() == "pads" )
			pads_attribute = parse_attribute_ints(a);
		else if( a.name() == "value" )
			value_attribute = parse_attribute_float(a);
		else
			LOG(FATAL) << "Ignoring attribute " << a.name() << " for node Pad/" << onnx_name << std::endl;
	}
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void Pad::resolve(void)
{
	const Tensor *data = get_input_tensor(0);
	name_input(0, "data");

	const Tensor *pads_tensor = nullptr;
	if (get_number_of_inputs() > 1) {
		pads_tensor = get_input_tensor(1);
		name_input(1, "pads");
	}
	const Tensor *constant_value= nullptr;
	if (get_number_of_inputs() > 2) {
		// This is not a tensor but a scalar. Not sure how to handle - first scalar in onnx2c :)
		constant_value = get_input_tensor(2);
		name_input(2, "constant_value");
	}

	if (pads_tensor && pads_tensor->isConst == false)
		ERROR("Non-constant 'pads' input to Pad would result in dynamic memory allocation");
	if( pads_tensor && pads_tensor->data_type != onnx::TensorProto_DataType_INT64 )
		ERROR("Malformed input. Input 2 to Pads is not a tensor of int64");

	// Use attribute is given, use that. The tensor should not be given in that case.
	constant = value_attribute;
	if (constant_value) {
		// The documentation of Pad is a bit complex, and it seems there are
		// .onnx generators out ther who create a tensor of undefined data type for Pad
		// This must be the default of pad with zeros they mean?
		if( constant_value->data_type == onnx::TensorProto_DataType_UNDEFINED ) {
			constant = 0;
		}
		else if (constant_value->isConst == false) {
			ERROR("Non-constant 'constant_value' input to Pad would result in dynamic memory allocation");
		}
		else {
			// Not sure this works. constant_value is supposed to be a scalar
			constant = constant_value->get_data_element_float(0);
		}
	}

	if (pads_tensor) {
		int i;
		for( i=0; i<pads_tensor->data_dim[0]/2; i++)
			paddings_start.push_back(pads_tensor->get_data_element(i));
		for( ; i<pads_tensor->data_dim[0]; i++)
			paddings_end.push_back(pads_tensor->get_data_element(i));
	}
	else {
		unsigned i;
		for( i=0; i<pads_attribute.size()/2; i++)
			paddings_start.push_back(pads_attribute[i]);
		for( ; i<pads_attribute.size(); i++)
			paddings_end.push_back(pads_attribute[i]);
	}

	/* Create output tensors).
	 * Set data dimensions and data type for the created tensors. */
	Tensor *t = new Tensor;
	for( unsigned d = 0; d < data->rank(); d++)
		t->data_dim.push_back( data->data_dim[d] + paddings_start[d] + paddings_end[d] );
	t->data_type = onnx::TensorProto_DataType_FLOAT;
	/* Store the created tensor both as reference in this node, and into
	 * the return value vector! */
	register_output(t, "output");

	/* TODO: optional outputs? */
}



/* Body of the node implementing function */
void Pad::print(std::ostream &dst) const
{
	INDT_1 << "/* Pad: " << std::endl;
	INDT_1 << " * pad at start: ";
	for( auto p : paddings_start )
		dst << p << " ";
	dst << std::endl;
	INDT_1 << " * pad at end:   ";
	for( auto p : paddings_end )
		dst << p << " ";
	dst << std::endl;
	INDT_1 << " * mode: " << mode << std::endl;
	INDT_1 << " */" << std::endl;

	const Tensor *data = get_input_tensor(0);
	const Tensor *output = get_output_tensor(0);

	std::string iidxs = "";
	std::string oidxs = "";
	for( unsigned i = 0; i<data->data_dim.size(); i++) {
		iidxs += "[ir" + std::to_string(i) + "]";
		oidxs += "[o" + std::to_string(i) + "]";
	}

	// Loop over all dimensions
	for( unsigned i = 0; i<data->data_dim.size(); i++) {
		std::string ilidx = "il" + std::to_string(i);
		std::string iridx = "ir" + std::to_string(i);
		std::string oidx = "o" + std::to_string(i);
		std::string dopad = "pad_at_" + std::to_string(i);

		// Print the loop in C source
		INDT(i+1) << "size_t " << iridx << ";" << std::endl;
		INDT(i+1) << "for( size_t " << oidx << "=0, " << ilidx << "=0; ";
		dst <<            oidx << "<" << output->data_dim[i] << "; ";
		dst <<            oidx <<"++ ) {" << std::endl;
		if ( mode == "constant" )
		{
			INDT(i+2) << "bool " << dopad << "=false;" << std::endl;
		}

		// Handle padding at the 'start' end
		INDT(i+2) << "if( " << oidx << " < " << paddings_start[i] << "){" << std::endl;
			if( mode == "reflect" ) {
				INDT(i+3) << iridx << "= " << paddings_start[i] << " - " << oidx << ";" << std::endl;
			}
			else if( mode == "edge" ) {
				INDT(i+3) << iridx << "= 0;" << std::endl;
			}
			else {  // constant
				INDT(i+3) << dopad << "= true;" << std::endl;
			}
		INDT(i+2) << "}" << std::endl;


		// Handle the non-padding copying
		INDT(i+2) << "else if( " << oidx << " < " << paddings_start[i]+data->data_dim[i] << "){" << std::endl;
			INDT(i+3) << iridx << "=" << ilidx << ";" << std::endl;
			INDT(i+3) << ilidx << "++;" << std::endl;
		INDT(i+2) << "}" << std::endl;

		// Handle padding at the end
		INDT(i+2) << "else {" << std::endl;
			if( mode == "reflect" ) {
				INDT(i+3) << iridx << " = 2*"<< data->data_dim[i] << "+" << paddings_start[i];
				dst << "-" << oidx << "-2;" << std::endl;
			}
			else if( mode == "edge" ) {
				INDT(i+3) << iridx << " = " << data->data_dim[i] << "-1;" << std::endl;
			}
			else  { // constant
				INDT(i+3) << dopad << "= true;" << std::endl;
			}
		INDT(i+2) << "}" << std::endl;
	}

	// We are now in the innermost loop: set single element output value
	if( mode == "constant" ) {
		INDT_1 << "if ( pad_at_0 ";
		for( unsigned i = 1; i<data->data_dim.size(); i++) {
		   dst << " || pad_at_" << std::to_string(i);
		}
		dst << ")" << std::endl;
		INDT_2 << "output" << oidxs << " = " << constant << ";" << std::endl;
		INDT_1 << "else" << std::endl;
	}
	INDT_2 << "output" << oidxs << "= data"<< iidxs << ";" << std::endl;

	// close loops
	for( unsigned i=data->data_dim.size(); i>0 ; i--) {
		INDT(i) << "}" << std::endl;
	}

}

} // namespace

