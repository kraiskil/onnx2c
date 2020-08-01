#pragma once
#include <string>
#include "error.h"
#include "onnx.pb.h"
#include "util.h"

namespace toC {

// TODO: artificial limitation from implementation. Check what ONNX says,
// and see if we already can handle that. 
// Uh - brainfart. Use std::vector<int> data_dim;
constexpr int tensor_max_dim=2;

// A entity that implements ONNX graph edges,
// i.e. the data buffers a ONNX node produces or consumes
class Tensor {
	public:
	bool generate; // generate code for this Tensor? (false for inputs)
	bool initialize; // generate initialization from data in data_buffer
	int data_dim[tensor_max_dim]; //zero if dimension not in use
	int data_num_elem;
	onnx::TensorProto_DataType data_type;
	void *data_buffer;

	std::string name; // NB: ONNX name. Might not be valid for C
	std::string doc;

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name sincce ONNX allows tensors and nodes
	 * to have the same name */
	std::string cname(void) const
	{
		return "tensor_" + cify_name(name);
	}

	int data_elem_size(void)const
	{
		switch( data_type )
		{
			case onnx::TensorProto_DataType_FLOAT:
				return sizeof(float); break;
			case onnx::TensorProto_DataType_INT32:
				return sizeof(int32_t); break;
			default:
				ERROR("unhandled tensor data type in tensor " << name);
				break;
		};
	}

	std::string data_type_str(void) const
	{
		switch( data_type )
		{
			case onnx::TensorProto_DataType_FLOAT:
				return "float"; break;
			case onnx::TensorProto_DataType_INT32:
				return "int32_t"; break;
			default:
				ERROR("unhandled tensor data type in tensor " << name);
				break;
		};
	}
};

}

