#pragma once
#include "error.h"
#include "onnx.pb.h"
#include <cstddef>
#include <string>

namespace toC {

class Node;

enum class TensorStorageKind {
	Dedicated,
	Union,
	Arena,
};

// A entity that implements ONNX graph edges,
// i.e. the data buffers a ONNX node produces or consumes
class Tensor {
	public:
	bool generate;    // generate code (i.e global definition) for this Tensor
	bool initialize;  // generate initialization from data in data_buffer
	bool isConst;     // constant value. Value is known at 'resolve()' time.
	bool isIO;        // is a parameter passed to the entry function of the graph.
	                  // IO tensors still get initialized e.g. in the test suite
	bool isRecursive; // tensor that one node uses both output and input.
	                  // may additionally be used as input for other nodes
	std::vector<int> data_dim;
	onnx::TensorProto_DataType data_type;
	void* data_buffer; // if initialized, contains the initialization data
	std::string name;  // NB: ONNX name. Might not be valid for C
	std::string doc;

	std::vector<Node*> consumers;
	Node* producer;
	TensorStorageKind storage_kind;
	int32_t union_no; // negative for no union
	size_t arena_offset;
	size_t arena_size;
	size_t arena_alignment;

	Tensor() : generate(true),
	           initialize(false),
	           isConst(false),
	           isIO(false),
	           isRecursive(false),
	           data_buffer(NULL),
	           producer(nullptr),
	           storage_kind(TensorStorageKind::Dedicated),
	           union_no(-1),
	           arena_offset(0),
	           arena_size(0),
	           arena_alignment(1)
	{
	}

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name sincce ONNX allows tensors and nodes
	 * to have the same name */
	std::string cname(void) const;

	/* Number of bytes of one data element */
	int data_elem_size(void) const;

	/* Total tensor storage size in bytes with overflow checks. */
	size_t data_size_bytes(void) const;

	/* Required alignment for typed C access. */
	size_t required_alignment(void) const;

	/* Number of elements in data.
	 * I.e. the product of the data dimensions */
	int data_num_elem(void) const;

	/* Number of data dimensions.
	 * Is zero for scalars*/
	unsigned rank(void) const;
	bool is_scalar(void) const { return rank() == 0; }

	bool eligible_for_arena(void) const;

	/* A string with the the C type for this tensor's data element. E.g. "float" */
	std::string data_type_str(void) const;

	/* Get the min and max values for this tensor's data type */
	std::pair<std::string, std::string> get_type_bounds() const;

	/* Fill this Tensor from the ONNX TensorProto */
	/* TODO: would this not be nicer as a constructor? :) */
	void parse_onnx_tensor(const onnx::TensorProto& tensor);

	/* Print the 'float foo[N][N]' part of the tensor.
	 * This is used to print out tensor initializers, parameters to function calls, and
	 * parameters in function definitions.
	 *
	 * If alternate_name is given, use that instead of the tensor's cname(),
	 * If not a callsite, print as a 'const' tensor if asConst.
	 * This is intended to print the tensors in a function declaration, definition and callsites.
	 * If callsite is true, skip the "float" and "[N][N]" parts.
	 */
	std::string print_tensor(
	    std::string alternate_name = "",
	    bool is_callsite = false,
	    bool as_const = false,
	    bool is_definition = false) const;
	std::string print_tensor_callsite(void) const
	{
		return print_tensor("", true, false);
	}
	std::string print_tensor_callsite_const(void) const;
	std::string arena_storage_member(void) const;
	std::string print_tensor_as_const(std::string alternate_name = "") const
	{
		return print_tensor(alternate_name, false, true);
	}
	std::string print_tensor_definition(std::string alternate_name = "") const
	{
		return print_tensor(alternate_name, false, false, true);
	}
	std::string print_arena_alias(void) const;

	/* Print a tensor's initialization to output stream.
	 * i.e. everything after the "=" in "float foo[43] = { 42, 42, ... };"
	 * Do not override dim and offs - used only by the function when it recurses into itself. */
	void print_tensor_initializer(std::ostream& destination, int dim = 0, int offs = 0) const;

	/* Print the i:th element in data_buffer */
	void print_element(std::ostream& dst, uint64_t i) const;

	/* Format dimensions into a string */
	std::string str_dimensions(void) const;

	/* Node definitions include the concept of optional inputs/outputs.
	 * This function tells wether a given tensor must be included or if it can be left out.
	 * This will return valid data only after all nodes have been resolved! (I.e. use it during printout phase)
	 */
	bool is_used(void) const;

	/* Get the data element at index i. Flattening multidimensional arrays down to the index is left for the caller. */
	int64_t get_data_element(uint64_t i) const;
	float get_data_element_float(uint64_t i) const;

	void clear_storage(void)
	{
		storage_kind = TensorStorageKind::Dedicated;
		union_no = -1;
		arena_offset = 0;
		arena_size = 0;
		arena_alignment = 1;
	}

	void assign_union(uint32_t u)
	{
		LOG(DEBUG) << "Assigning tensor " << cname() << " to union " << u << std::endl;
		storage_kind = TensorStorageKind::Union;
		union_no = u;
	}

	void assign_arena(size_t offset, size_t size, size_t alignment)
	{
		storage_kind = TensorStorageKind::Arena;
		union_no = -1;
		arena_offset = offset;
		arena_size = size;
		arena_alignment = alignment;
	}

	std::string print_trace_dump(void) const;
};

} // namespace toC
