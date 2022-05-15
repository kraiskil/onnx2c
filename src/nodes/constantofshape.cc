/* This file is part of onnx2c.
 *
 * ConstantOfShape node.
 */
#include "constantofshape.h"
using namespace toC;

void ConstantOfShape::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "value" )
			value = parse_attribute_tensor(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node ConstantOfShape/" << onnx_name << std::endl;
	}
}


void ConstantOfShape::resolve(void)
{
	input  = inputs[0];
	register_input(input, "input");

	Tensor *t = new Tensor;
	for( int i=0; i<input->data_num_elem(); i++) {
		uint64_t d = input->get_data_element(i);
		t->data_dim.push_back(d);
	}

	if( value )
		t->data_type = value->data_type;
	else
		t->data_type = onnx::TensorProto_DataType_FLOAT;
	output = t;
	outputs.push_back(t);
	register_output(t, "output");
}


/* Body of the node implementing function */
void ConstantOfShape::print(std::ostream &dst) const
{
	INDT_1 << "/* ConstantOfShape */" << std::endl;
	std::string type = output->data_type_str();

	INDT_1 << type << " *dst = (" << type << "*)output;" << std::endl;
	INDT_1 << "for( unsigned i=0; i< " << output->data_num_elem() << "; i++)" << std::endl;
	INDT_2 <<   "dst[i] = " ;
	if( value == NULL )
		dst << "0;" <<std::endl;
	else if( isFloat( value->data_type ) )
		dst << value->get_data_element_float(0) << ";" <<std::endl;
	else
		dst << value->get_data_element(0) << ";" <<std::endl;
}


