#pragma once
#include <string>
#include "error.h"
#include "onnx.pb.h"

namespace toC {


// A entity that implements ONNX graph edges,
// i.e. the data buffers a ONNX node produces or consumes
class Tensor {
	public:
	bool generate; // generate code for this Tensor? (false for inputs)
	bool initialize; // generate initialization from data in data_buffer
	std::vector<int> data_dim;
	int data_num_elem;
	onnx::TensorProto_DataType data_type;
	void *data_buffer;

	std::string name; // NB: ONNX name. Might not be valid for C
	std::string doc;

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name sincce ONNX allows tensors and nodes
	 * to have the same name */
	std::string cname(void) const;

	/* Number of bytes of one data element */
	int data_elem_size(void)const;

	/* A string with the the C type for this tensor's data element. E.g. "float" */
	std::string data_type_str(void) const;

	/* Fill this Tensor from the ONNX TensorProto */
	void parse_onnx_tensor(const onnx::TensorProto &tensor);


	/* Print the 'float foo[N][N]' part of the tensor.
	 * Optionally, prefix name with given prefix: used for the test suite. */
	void print_type_name_dimensions(std::ostream &destination, std::string prefix = "");

	/* Print a tensor's initialization to output stream.
	 * i.e. everything after the "=" in "float foo[43] = { 42, 42, ... };"
	 * Do not override dim and offs - used only by the function when it recurses into itself. */
	void print_tensor_initializer(std::ostream &destination, int dim=0, int offs=0);
};

}

