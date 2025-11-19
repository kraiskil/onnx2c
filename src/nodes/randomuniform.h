/* This file is part of onnx2c.
 *
 * RandomUniform node.
 *
 */

#include "node.h"

namespace toC {

class RandomUniform : public Node {
	public:
	RandomUniform()
	{
		op_name = "RandomUniform";
		dtype = 1;
		high = 1;
		low = 0;
	}

	// Attributes
	int dtype;
	float high;
	float low;
	std::vector<int> shape;

	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

void RandomUniform::parseAttributes(onnx::NodeProto& node)
{
	for (const auto& a : node.attribute()) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if (a.name() == "dtype")
			dtype = parse_attribute_int(a);
		else if (a.name() == "high")
			high = parse_attribute_float(a);
		else if (a.name() == "low")
			low = parse_attribute_float(a);
		else if (a.name() == "shape") {
			shape.clear();
			for (int64_t dim : parse_attribute_ints(a))
				shape.push_back((int)dim);
		}
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node RandomUniform/" << onnx_name << std::endl;
	}
}

void RandomUniform::resolve(void)
{
	Tensor* t = new Tensor;
	t->data_dim = shape;
	t->data_type = (onnx::TensorProto_DataType)dtype;
	register_output(t, "output");
}

void RandomUniform::print(std::ostream& dst) const
{
	INDT_1 << "/* RandomUniform */" << std::endl;

	Tensor* output = get_output_tensor(0);

	INDT_1 << output->data_type_str() << " *output_ptr = (" << output->data_type_str() << "*) output;" << std::endl;
	INDT_1 << "for( int i=0; i<" << output->data_num_elem() << "; i++ )" << std::endl;
	INDT_2 << "output_ptr[i] = (" << output->data_type_str() << ")"
	       << "(rand() / (float) RAND_MAX) * " << (high - low) << " + " << low << ";" << std::endl;
}

} // namespace toC
