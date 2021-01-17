#pragma once
#include <string>
#include "error.h"
#include "onnx.pb.h"

namespace toC {


// A entity that implements ONNX graph edges,
// i.e. the data buffers a ONNX node produces or consumes
class Tensor {
	public:
	bool generate;   // generate code (i.e global definition) for this Tensor
	bool initialize; // generate initialization from data in data_buffer
	bool isConst;    // constant value.
	bool isIO;       // is a parameter passed to the entry function of the graph.
	                 // IO tensors still get initialized e.g. in the test suite
	bool isRecursive;// tensor that one node uses both output and input.
	                 // may additionally be used as input for other nodes
	const Tensor *isAliasOf; // Recursive tensors might (but need not) be defined
	                         // twice: as the input and the output.
	std::vector<int> data_dim;
	onnx::TensorProto_DataType data_type;
	void *data_buffer;// if initialized, contains the initialization data
	std::string name; // NB: ONNX name. Might not be valid for C
	std::string doc;

	Tensor() :
		generate(true),
		initialize(false),
		isConst(true),
		isIO(false),
		isRecursive(false),
		isAliasOf(NULL),
		data_buffer(NULL)
	{}

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name sincce ONNX allows tensors and nodes
	 * to have the same name */
	std::string cname(void) const;

	/* Number of bytes of one data element */
	int data_elem_size(void)const;

	/* Number of elements in data.
	 * I.e. the product of the data dimensions */
	int data_num_elem(void) const;

	/* Number of data dimensions */
	unsigned rank(void) const;

	/* A string with the the C type for this tensor's data element. E.g. "float" */
	std::string data_type_str(void) const;

	/* Fill this Tensor from the ONNX TensorProto */
	void parse_onnx_tensor(const onnx::TensorProto &tensor);


	/* Print the 'float foo[N][N]' part of the tensor.
	 * If anternate_name is given, use that instead of 'foo',
	 * This is intended to print the tensors in a function declaration, definition and callsites.
	 * If callsite is true, skip the "float" and "[N][N]" parts.
	 */
	void print_tensor(std::ostream &destination, bool callsite=false, std::string alternate_name = "") const;

	/* Print a tensor's initialization to output stream.
	 * i.e. everything after the "=" in "float foo[43] = { 42, 42, ... };"
	 * Do not override dim and offs - used only by the function when it recurses into itself. */
	void print_tensor_initializer(std::ostream &destination, int dim=0, int offs=0);

	/* Print the i:th element in data_buffer */
	void print_element(std::ostream &dst, uint64_t i) const;

	/* Format dimensions into a string */
	std::string str_dimensions(void);
};

}

