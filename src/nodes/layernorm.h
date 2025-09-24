/* This file is part of onnx2c.
 *
 * LayerNormalization node.
 * 
 */

#include "node.h"

namespace toC {

class LayerNormalization : public Node {
	public:
	LayerNormalization() {
		op_name = "LayerNormalization";
		axis = -1;
		epsilon = 1e-5;
		stash_type = 1;
	}

	// Attributes
	int axis;
	float epsilon;
	int stash_type;

	virtual void parseAttributes(onnx::NodeProto &node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};


void LayerNormalization::parseAttributes( onnx::NodeProto &node ) {
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "axis" )
			axis = parse_attribute_int(a);
		else if( a.name() == "epsilon" )
			epsilon = parse_attribute_float(a);
		else if( a.name() == "stash_type" )
			stash_type = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node LayerNormalization/" << onnx_name << std::endl;
	}
}

void LayerNormalization::resolve(void) {
	Tensor *x = get_input_tensor(0);
	
	name_input(0, "x");
	name_input(1, "scale");
	if (get_number_of_inputs() == 3) {
		name_input(2, "b");
	}

	if (axis < 0) {
		axis += x->data_dim.size();
	}

	assert(stash_type == 1);

	Tensor *y = new Tensor;
	y->data_dim = x->data_dim;
	y->data_type = x->data_type;
	register_output(y, "y");

	Tensor *mean = new Tensor;
	mean->data_dim = std::vector<int>(x->data_dim.begin(), x->data_dim.begin() + axis);
	mean->data_dim.push_back(1);
	mean->data_type = onnx::TensorProto_DataType_FLOAT;
	register_output(mean, "mean");

	// Same as mean
	Tensor *inv_std_dev = new Tensor;
	inv_std_dev->data_dim = mean->data_dim;
	inv_std_dev->data_type = mean->data_type;
	register_output(inv_std_dev, "inv_std_dev");
}

static std::string broadcast(Tensor *t, const std::string &name, int to_rank) {
	if (t->is_scalar()) {
		return "*" + name;	
	}
	std::ostringstream dst;
	dst << name;
	for (int i = 0; i < (int)t->data_dim.size(); i++) {
		dst << "[i" << (to_rank - t->data_dim.size() + i) << "]";
	}
	return dst.str();
}

void LayerNormalization::print(std::ostream &dst) const {
	INDT_1 << "/* LayerNormalization */" << std::endl;

	Tensor *x = get_input_tensor(0);
	Tensor *scale = get_input_tensor(1);

	std::string stash_type_str = "float";

	std::string outer_idx = "";
	for (int i = 0; i < axis; i++) {
		INDT_1 << "for (unsigned i" << i << " = 0; i" << i << "<" << x->data_dim[i] << "; i" << i << "++)" << std::endl;
		outer_idx += "[i" + std::to_string(i) + "]";
	}

	INDT_1 << "{" << std::endl;

	std::string idx = outer_idx;
	int inner_element_count = 1;
	for (int i = axis; i < (int)x->data_dim.size(); i++) {
		idx += "[i" + std::to_string(i) + "]";
		inner_element_count *= x->data_dim[i];
	}

	// Compute mean
	INDT_2 << stash_type_str << " mean_value = 0;" << std::endl;
	for (int i = axis; i < (int)x->data_dim.size(); i++) {
		INDT_2 << "for (unsigned i" << i << " = 0; i" << i << "<" << x->data_dim[i] << "; i" << i << "++)" << std::endl;
	}

	INDT_3 << "mean_value += x" << idx << " / (" << stash_type_str << ")" << inner_element_count << ";" << std::endl;
	
	// Compute variance
	INDT_2 << stash_type_str << " variance_value = 0;" << std::endl;
	for (int i = axis; i < (int)x->data_dim.size(); i++) {
		INDT_2 << "for (unsigned i" << i << " = 0; i" << i << "<" << x->data_dim[i] << "; i" << i << "++)" << std::endl;
	}
	INDT_3 << "variance_value += (x" << idx << " - mean_value) * (x" << idx << " - mean_value) / (" << stash_type_str << ")" << inner_element_count << ";" << std::endl;

	INDT_2 << stash_type_str << " inv_std_dev_value = 1.0f / sqrtf(variance_value + " << epsilon << "f);" << std::endl;

	// Normalize
	for (int i = axis; i < (int)x->data_dim.size(); i++) {
		INDT_2 << "for (unsigned i" << i << " = 0; i" << i << "<" << x->data_dim[i] << "; i" << i << "++)" << std::endl;
	}
	INDT_3 << "y" << idx << " = (x" << idx << " - mean_value) * inv_std_dev_value * " << broadcast(scale, "scale", x->rank());
	if (get_number_of_inputs() == 3) {
		Tensor *b = get_input_tensor(2);
		dst << " + " << broadcast(b, "b", x->rank());
	}
	dst << ";" << std::endl;

	// Store mean and inv_std_dev
	INDT_2 << "mean" << outer_idx << "[0] = mean_value;" << std::endl;
	INDT_2 << "inv_std_dev" << outer_idx << "[0] = inv_std_dev_value;" << std::endl;

	INDT_1 << "}" << std::endl;
}

} // namespace

