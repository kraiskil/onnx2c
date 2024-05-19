/* This file is part of onnx2c.
 *
 * Cast node. Casts between float/double/string/half floats.
 */
#include "cast.h"

namespace toC {

void Cast::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "to" )
			to = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node Cast/" << onnx_name << std::endl;
	}
}


void Cast::resolve(void)
{
	LOG(INFO) << "'Cast' node found." << std::endl;
	name_input(0, "input");

	switch(to)
	{
		case onnx::TensorProto_DataType_INT32:
			output_type = "int32_t"; break;
		case onnx::TensorProto_DataType_FLOAT:
			output_type = "float";   break;
		case onnx::TensorProto_DataType_DOUBLE:
			output_type = "double";  break;
		default:
			ERROR("Unimplemented casting to requested type");
	}

	Tensor *t = new Tensor;
	t->data_dim = get_input_tensor(0)->data_dim;
	t->data_type = static_cast<onnx::TensorProto_DataType>(to);
	register_output(t, "output");
}


void Cast::print(std::ostream &dst) const
{
	INDT_1 << "/* Cast */" << std::endl;
	const Tensor *input = get_input_tensor(0);
	const Tensor *output = get_output_tensor(0);

	std::string intype = input->data_type_str();
	std::string outtype = output->data_type_str();

	INDT_1 << intype << " *X = (" << intype << "*)input;" << std::endl;
	INDT_1 << outtype << " *Y = (" << outtype << "*)output;" << std::endl;

	INDT_1 << "for( unsigned i=0; i<"<<input->data_num_elem() << "; i++)"  << std::endl;
	INDT_2 <<   "Y[i]= (" << outtype << ")X[i];" << std::endl;
}

} // namespace

