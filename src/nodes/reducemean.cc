/* This file is part of onnx2c.
 *
 * ReduceMean node.
 *
 */
#include "reducemean.h"

#include <algorithm>

namespace toC {

void ReduceMean::parseAttributes( onnx::NodeProto &node )
{
	if (onnx_ir_version > 13)
	{
		// At version 18, the attributes changed.
		ERROR("ReduceMean only supported for opset <=13 not " + std::to_string(onnx_ir_version));
	}

	Tensor* input = get_input_tensor(0);

	// Default to reducing all dimensions
	for (int i = 0; i < input->data_dim.size(); ++i)
	{
		axes.push_back(i);
	}

	// Default to keeping all dimensions
	keepdims = 1;

	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "axes" )
		{
			axes = parse_attribute_ints(a);

			for (int64_t& axis : axes)
			{
				while (axis < 0)
				{
					axis += get_input_tensor(0)->data_dim.size();
				}
			}
		}
		else if( a.name() == "keepdims" )
			keepdims = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node ReduceMean/" << onnx_name << std::endl;
	}
}

/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void ReduceMean::resolve(void)
{
	name_input(0, kInputName);
	Tensor *input = get_input_tensor(0);

	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
	Tensor *t = new Tensor;
	t->data_dim = {};

	for (int i = 0; i < input->data_dim.size(); ++i)
	{
		if(std::find(axes.begin(), axes.end(), i) != axes.end())
		{
			if (keepdims)
			{
				t->data_dim.push_back(1);
			}
			continue;
		}

		t->data_dim.push_back(input->data_dim[i]);
	}

	t->data_type = input->data_type;
	register_output(t, kOutputName);
}

/* Body of the node implementing function */
void ReduceMean::print(std::ostream &dst) const
{
	const Tensor *input = get_input_tensor(0);
	std::string datatype = input->data_type_str();

	const Tensor *output = get_output_tensor(0);

	INDT_1 << "/* ReduceMean */" << std::endl;

	// Allocate arrays for intermediate sums/means (final reduction goes directly to output)
	std::vector<int> sumDims = input->data_dim; // used to track axis reductions based on the keepdims attribute
	std::unordered_map<int64_t, std::vector<int>> axisDims; // dimensions at each reduction
	for (int i = 0; i < axes.size() - 1; ++i)
	{
		int64_t axis = axes[i];
		INDT_1 << input->data_type_str() << " sum_axis_" << axis;
		sumDims[axis] = keepdims ? 1 : 0;
		for (int i = 0; i < sumDims.size(); ++i)
		{
			if (sumDims[i])
			{
				dst << "[" << sumDims[i] << "]";
				axisDims[axis].push_back(sumDims[i]);
			}
		}
		dst << " = {0};" << std::endl;
	}

	// Final reduction goes to the output tensor
	axisDims[axes[axes.size() - 1]] = get_output_tensor(0)->data_dim;

	// Store dimensions of tensors (including output)
	for (int64_t axis : axes)
	{
		INDT_1 << "int dims_" << axis << "[" << axisDims[axis].size() << "] = {";
		for (int i = 0; i < axisDims[axis].size(); ++i)
		{
			dst << axisDims[axis][i];
			if (i < axisDims[axis].size() - 1)
			{
				dst << ", ";
			}
		}
		dst << "};" << std::endl;
	}

	// Set output to 0 like the intermediate sums
	int totalOutputElements = 1;
	for (int dim : output->data_dim)
	{
		totalOutputElements *= dim;
	}
	INDT_1 << "for (int j = 0; j < " << totalOutputElements << "; ++j) {" << std::endl;
		
	// Tracking sum index
	printLocationArray(dst, 2, axes[axes.size() - 1], output->data_dim.size(), "j");

	// Set to 0
	INDT_2 << kOutputName;
	for (int i = 0; i < output->data_dim.size(); ++i)
	{
		dst << "[loc[" << i << "]]";
	}
	dst << " = 0;" << std::endl;

	INDT_1 << "}" << std::endl;

	// Reduce each requested axis
	std::string prevTensorName = kInputName; // remember previous tensor to use for next sum
	for (int i = 0; i < axes.size(); ++i)
	{
		// Comment the axis being reduced
		INDT_1 << "// Reduce axis " << axes[i] << std::endl;

		// Step 1: Get sum
		int64_t axis = axes[i];
		std::string outputTensor = (i == axes.size() - 1) ? kOutputName : ("sum_axis_" + std::to_string(axis));
		INDT_1 << "for (int i = 0; i < " << input->data_dim[axis] << "; ++i) {" << std::endl;

		int totalElements = 1;
		std::vector<int>& currentAxisDims = axisDims[axis];
		for (int dim : currentAxisDims)
		{
			totalElements *= dim;
		}
		INDT_2 << "for (int j = 0; j < " << totalElements << "; ++j) {" << std::endl;

		// Tracking sum index
		printLocationArray(dst, 3, axis, currentAxisDims.size(), "j");

		// Add to sum
		INDT_3 << outputTensor;
		for (int i = 0; i < currentAxisDims.size(); ++i)
		{
			dst << "[loc[" << i << "]]";
		}
		dst << " += " << prevTensorName;
		// Get dimensions of tensor before reduction
		const std::vector<int>& beforeReductionDims = (i == 0) ? input->data_dim : axisDims[axes[i - 1]];
		for (int i = 0, locIndex = 0; i < beforeReductionDims.size(); ++i)
		{
			if (i == axis)
			{
				dst << "[i]";

				if (keepdims)
				{
					locIndex++;
				}
			}
			else
			{
				dst << "[loc[" << locIndex++ << "]]";
			}
		}
		if (axis == currentAxisDims.size())
		{
			dst << "[i]";
		}
		dst << ";" << std::endl;

		INDT_2 << "}" << std::endl;

		INDT_1 << "}" << std::endl;

		// Step 2: Divide the sum to get the mean
		INDT_1 << "for (int j = 0; j < " << totalElements << "; ++j) {" << std::endl;

		// Tracking sum index
		printLocationArray(dst, 2, axis, currentAxisDims.size(), "j");

		// Convert sum to mean
		INDT_2 << outputTensor;
		for (int i = 0; i < currentAxisDims.size(); ++i)
		{
			dst << "[loc[" << i << "]]";
		}
		dst << " /= " << input->data_dim[axis] << ";" << std::endl;

		INDT_1 << "}" << std::endl;

		prevTensorName = "sum_axis_" + std::to_string(axis);
	}
}

// Creates an array called loc that stores the location in the tensor converted from a flat index
void ReduceMean::printLocationArray(std::ostream &dst, int indent, int axis, int dims, const char* flatIndexVariable) const
{
	INDT(indent) << "int loc[" << dims << "] = {";
	for (int i = 0; i < dims; ++i)
	{
		dst << "0";
		if (i < dims - 1)
		{
			dst << ", ";
		}
	}
	dst << "};" << std::endl;

	INDT(indent) << "int tempTotal = " << flatIndexVariable << ";" << std::endl;

	INDT(indent) << "for (int locIndex = " << dims - 1 << "; locIndex >= 0; --locIndex) {" << std::endl;

	INDT(indent + 1) << "loc[locIndex] = tempTotal % dims_" << axis << "[locIndex];" << std::endl;
	INDT(indent + 1) << "tempTotal /= dims_" << axis << "[locIndex];" << std::endl;

	INDT(indent) << "}" << std::endl;
}


} // namespace
