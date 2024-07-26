/* This file is part of onnx2c.
 *
 * Split.
 * "Split a tensor into a list of tensors, along the specified ‘axis’."
 */

#include <numeric>

namespace toC {

class Split : public Node {
	public:
	Split() {
		op_name = "Split";
	}

	int64_t axis = 0;

	virtual void parseAttributes( onnx::NodeProto &node ) override
	{
		for( const auto& a : node.attribute() ) {
			if( a.name() == "axis" )
				axis = parse_attribute_int(a);
			else
				ERROR("Bad attribute " << a.name() << " to split");
		}
	}

	virtual void print(std::ostream &dst) const override
	{
		// TODO
		dst << "\t/*Split*/" << std::endl;
		dst << std::endl;
		ERROR("Print for 'Split' node is not yet implemented :(");

	}

	virtual void resolve(void) override
	{
		auto split_sum = 0;

		const Tensor *data = get_input_tensor(0);
		const Tensor *split = get_input_tensor(1);
		name_input(0, "input");
		name_input(1, "split");

		// TODO in v18 'num_outputs' is an attribute
		int64_t num_outputs = split->data_dim[0];

		for (int i = 0; i < split->data_num_elem(); i++)
		{
			auto e = split->get_data_element(i);
			if (e < 0)
			{
				ERROR("'split' values must be greater than zero");
			}
			split_sum += split->get_data_element(i);
		}

		size_t ax_idx;
		if (ax_idx < 0)
		{
			ax_idx = data->data_dim.size() + ax_idx - 1;
		}
		else
		{
			ax_idx = axis;
		}

		if (data->data_dim[ax_idx] != split_sum)
		{
			ERROR("Sum of 'split' values must be equal to the dim value at 'axis' parameter ("
				  << data->data_dim[ax_idx] << ")");
		}

		for (int64_t i = 0; i < num_outputs; i++)
		{
			Tensor *rv = new Tensor;

			rv->data_type = data->data_type;
			for(uint64_t j = 0; j < data->data_dim.size(); j++)
			{
				if(j == ax_idx)
				{
					rv->data_dim.push_back(split->get_data_element(i));
				} else {
					rv->data_dim.push_back(data->data_dim[j]);
				}
			}	
			register_output(rv, "output_" + std::to_string(i));
		}
	}
};
}
