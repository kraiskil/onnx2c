/* This file is part of onnx2c.
 *
 * QuantizeLinear node.
 * 
 */

#include "node.h"

namespace toC {

class QuantizeLinear : public Node {
	public:
	QuantizeLinear() {
		op_name = "QuantizeLinear";
		axis = 1;
	}

	// Attributes
	int axis;

	virtual void parseAttributes(onnx::NodeProto &node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};


void QuantizeLinear::parseAttributes( onnx::NodeProto &node ) {
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "axis" )
			axis = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node QuantizeLinear/" << onnx_name << std::endl;
	}
}

void QuantizeLinear::resolve(void) {
	Tensor *x = get_input_tensor(0);
	name_input(0, "x");
	name_input(1, "y_scale");

	if (axis < 0) {
		axis += x->data_dim.size();
	}

	onnx::TensorProto_DataType output_data_type = onnx::TensorProto_DataType_UINT8;
	if (get_number_of_inputs() == 3) {
		name_input(2, "y_zero_point");
		Tensor *y_zero_point = get_input_tensor(2);
		output_data_type = y_zero_point->data_type;
	}
	
	assert(
		output_data_type == onnx::TensorProto_DataType_INT8 ||
		output_data_type == onnx::TensorProto_DataType_UINT8 ||
		output_data_type == onnx::TensorProto_DataType_INT16 ||
		output_data_type == onnx::TensorProto_DataType_UINT16
	);

	Tensor *t = new Tensor;
	t->data_dim = x->data_dim;
	t->data_type = output_data_type;
	register_output(t, "y");
}

void QuantizeLinear::print(std::ostream &dst) const {
	INDT_1 << "/* QuantizeLinear */" << std::endl;

	Tensor *x = get_input_tensor(0);
	Tensor *y_scale = get_input_tensor(1);
	Tensor *y = get_output_tensor(0);

	std::string index;
	for (unsigned loop_axis = 0; loop_axis < x->rank(); loop_axis++) {
		std::string name = "i" + std::to_string(loop_axis);
		INDT_1 << "for (unsigned " << name << " = 0; " << name << " < " << x->data_dim[loop_axis] << "; " << name << "++)" << std::endl;

		index += "[" + name + "]";
	}

	std::string param_index;
	assert(y_scale->data_dim.size() == 1);
	if (y_scale->data_dim[0] == 1) {
		param_index = "[0]";
	} else {
		param_index = "[i" + std::to_string(axis) + "]";
	}

	INDT_1 << "{" << std::endl;
	// Transform
	// We assume that roundf rounds .5 to nearest even
	INDT_2 << "int t = (int)roundf(x" << index << " / y_scale" << param_index << ")";
	if (get_number_of_inputs() == 3) {
		dst << " + (int)y_zero_point" << param_index;
	}
	dst << ";" << std::endl;
	// Saturate
	int lower, upper;
	switch (y->data_type) {
		case onnx::TensorProto_DataType_INT8:
			lower = -128;
			upper = 127;
			break;
		case onnx::TensorProto_DataType_UINT8:
			lower = 0;
			upper = 255;
			break;
		case onnx::TensorProto_DataType_INT16:
			lower = -32768;
			upper = 32767;
			break;
		case onnx::TensorProto_DataType_UINT16:
			lower = 0;
			upper = 65535;
			break;
		default:
			ERROR("Unsupported output data type for QuantizeLinear");
			break;
	}
	INDT_2 << "y" << index << " = (" << y->data_type_str() << ")MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
	INDT_1 << "}" << std::endl;
}

} // namespace

