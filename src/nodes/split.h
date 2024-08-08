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
		const Tensor *input = get_input_tensor(0);
		const Tensor *split = get_input_tensor(1);
		int64_t num_outputs = split->data_dim[0];
		int64_t num_dims = input->data_dim.size();
		auto io_type_str = input->data_type_str();

		dst << "\t/*Split*/" << std::endl;
		dst << "\tconst size_t axis = " << std::to_string(axis) << ";" << std::endl;
		dst << "\tconst size_t dims[" << std::to_string(num_dims) << "] = {";

		for(int64_t i = 0; i < num_dims; i++)
		{
			int64_t dim = 1;
			for(int64_t j = i + 1; j < num_dims; j++)
			{
				dim *= input->data_dim[j];
			}
			dst << std::to_string(dim);

			if(i < num_dims - 1)
			{
				dst << ", ";
			}
		}

		dst << "};" << std::endl;
		dst << "\tsize_t idx[" << std::to_string(num_dims) << "];" << std::endl
			<< std::endl;

		dst << "\tfor (size_t i = 0; i < (";
		for (int64_t i = 0; i < num_dims; i++)
		{
			dst << std::to_string(input->data_dim[i]);
			if (i < num_dims - 1)
			{
				dst << " * ";
			}
		}

		dst << "); i++)" << std::endl;

		dst << "\t{" << std::endl;
		dst << "\t\tsize_t t = i;" << std::endl << std::endl;

		dst << "\t\tfor (size_t j = 0; j < " << std::to_string(num_dims) << "; j++)" << std::endl;
		dst << "\t\t{" << std::endl;
		dst << "\t\t\tidx[j] = t / dims[j];" << std::endl;
		dst << "\t\t\tt %= dims[j];" << std::endl;

		dst << "\t\t}" << std::endl << std::endl;
		dst << "\t\t"<< io_type_str << " x = ((" << io_type_str << " *)input)[i];" << std::endl;

		dst << "\t\tsize_t split_idx = idx[axis];" << std::endl;
		dst << "\t\tsize_t split_sum = 0;" << std::endl;
		dst << "\t\tsize_t out_idx;" << std::endl << std::endl;

		dst << "\t\tint64_t offset = 0;" << std::endl;
		dst << "\t\tfor (out_idx = 0; out_idx < " << num_outputs << "; out_idx++)" << std::endl;
		dst << "\t\t{" << std::endl;
		dst << "\t\t\tsplit_sum += split[out_idx];" << std::endl;
		dst << "\t\t\tif (split_idx < split_sum)" << std::endl;
		dst << "\t\t\t{" << std::endl;
		dst << "\t\t\t\tbreak;" << std::endl;
		dst << "\t\t\t}" << std::endl;
		dst << "\t\t\toffset += split[out_idx];" << std::endl;
		dst << "\t\t}" << std::endl << std::endl;

		dst << "\t\tswitch (out_idx)" << std::endl;
		dst << "\t\t{" << std::endl;

		for (int64_t i = 0; i < num_outputs; i++)
		{
			dst << "\t\t\tcase " << std::to_string(i) << ":" << std::endl;
			dst << "\t\t\t\toutput_" << std::to_string(i);
			for(int64_t j = 0; j < num_dims; j++)
			{
				dst << "[";
				if(j == axis)
				{
					dst << "idx[" << std::to_string(j) << "] - offset";
				} else {
					dst << "idx[" << std::to_string(j) << "]";
				}
				dst << "]";
			}
			dst << " = x;" << std::endl;
			dst << "\t\t\t\tbreak;" << std::endl << std::endl;
		}

		dst << "\t\t\tdefault:" << std::endl;
		dst << "\t\t\t\tbreak;" << std::endl;

		dst << "\t\t}" << std::endl << std::endl;

		dst << "\t}" << std::endl;

		dst << std::endl;
	}

	virtual void resolve(void) override
	{
		auto split_sum = 0;

		const Tensor *input = get_input_tensor(0);
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

		if (axis < 0)
		{
			axis = input->data_dim.size() + axis;
		}

		if (input->data_dim[axis] != split_sum)
		{
			ERROR("Sum of 'split' values must be equal to the dim value at 'axis' parameter ("
				  << input->data_dim[axis] << ")");
		}

		for (int64_t i = 0; i < num_outputs; i++)
		{
			Tensor *rv = new Tensor;

			rv->data_type = input->data_type;
			for(uint64_t j = 0; j < input->data_dim.size(); j++)
			{
				if(j == (uint64_t)axis)
				{
					rv->data_dim.push_back(split->get_data_element(i));
				} else {
					rv->data_dim.push_back(input->data_dim[j]);
				}
			}	
			register_output(rv, "output_" + std::to_string(i));
		}
	}
};
}
