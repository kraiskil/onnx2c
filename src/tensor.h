#pragma once
#include <string>
#include "error.h"
#include "onnx.pb.h"

namespace toC {

class Node;
// A entity that implements ONNX graph edges,
// i.e. the data buffers a ONNX node produces or consumes
class Tensor {
	public:
	bool generate;   // generate code (i.e global definition) for this Tensor
	bool initialize; // generate initialization from data in data_buffer
	bool isConst;    // constant value. Value is known at 'resolve()' time.
	bool isIO;       // is a parameter passed to the entry function of the graph.
	                 // IO tensors still get initialized e.g. in the test suite
	bool isRecursive;// tensor that one node uses both output and input.
	                 // may additionally be used as input for other nodes
	Tensor *quantizedCopy; // non-NULL if there is a quantized version of this
	bool isQuantized;  // is this a quantized copy
	std::vector<int> data_dim;
	onnx::TensorProto_DataType data_type;
	void *data_buffer;// if initialized, contains the initialization data
	std::string name; // NB: ONNX name. Might not be valid for C
	std::string doc;

	std::vector<Node *> consumers;
	int32_t union_no;     // negative for no union

	Tensor() :
		generate(true),
		initialize(false),
		isConst(false),
		isIO(false),
		isRecursive(false),
		quantizedCopy(NULL),
		isQuantized(false),
		data_buffer(NULL),
		union_no(-1)
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
	/* TODO: would this not be nicer as a constructor? :) */
	void parse_onnx_tensor(const onnx::TensorProto &tensor);


	/* Print the 'float foo[N][N]' part of the tensor.
	 * This is used to print out tensor initializers, parameters to function calls, and
	 * parameters in function definitions.
	 *
	 * If alternate_name is given, use that instead of the tensor's cname(),
	 * If not a callsite, print as a 'const' tensor if asConst.
	 * This is intended to print the tensors in a function declaration, definition and callsites.
	 * If callsite is true, skip the "float" and "[N][N]" parts.
	 */
	void print_tensor(std::ostream &destination, bool callsite=false, std::string alternate_name = "", bool asConst=false) const;
	/* Shortcut to previous */
	void print_tensor_as_const(std::ostream &destination, bool callsite=false, std::string alternate_name = "") const
	{
		print_tensor(destination, callsite, alternate_name, true);
	}

	/* Same as above, but return as string */
	std::string print_tensor(std::string alternate_name, bool is_callsite=false, bool as_const=false) const;
	std::string print_tensor_callsite(void) const
	{
		return print_tensor( "", true, false );
	}
	std::string print_tensor_as_const(std::string alternate_name) const
	{
		return print_tensor( alternate_name, false, true );
	}


	/* Print a tensor's initialization to output stream.
	 * i.e. everything after the "=" in "float foo[43] = { 42, 42, ... };"
	 * Do not override dim and offs - used only by the function when it recurses into itself. */
	void print_tensor_initializer(std::ostream &destination, int dim=0, int offs=0) const;

	/* Print the i:th element in data_buffer */
	void print_element(std::ostream &dst, uint64_t i) const;

	/* Format dimensions into a string */
	std::string str_dimensions(void) const;

	Tensor* make_quantized_copy(void);

	/* Node definitions include the concept of optional inputs/outputs.
	 * This function tells wether a given tensor must be included or if it can be left out.
	 * This will return valid data only after all nodes have been resolved! (I.e. use it during printout phase)
	 */
	bool is_used(void) const;

	/* Get the data element at index i. Flattening multidimensional arrays down to the index is left for the caller. */
	int64_t get_data_element(uint64_t i) const;
	float get_data_element_float(uint64_t i) const;


	void assign_union(uint32_t u) {
		LOG(DEBUG) << "Assigning tensor " << cname() << " to union " << u <<std::endl;
		union_no = u;
	}

	std::string print_trace_dump(void) const;
};

}

