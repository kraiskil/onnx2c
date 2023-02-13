#pragma once
#include <string>
#include <tuple>
#include "error.h"
#include "onnx.pb.h"
#include "util.h"

namespace toC {

class Tensor;
typedef std::tuple<Tensor *, std::string> function_parameter;

/* The ONNX node, or computation kernel. *
 * Node is a virtual parent class for each of the
 * ONNX node "types" or "operands" (e.g. Add, Relu, ...)
 * Each individual node in the graph is then an instance of
 * these subclasses.
 */
class Node {
	public:
	bool isResolved;       // has this node been visited in current compilation step.
	std::string onnx_name; //ONNX name of the individual node
	std::string op_name;   //ONNX name of node type
	static int64_t onnx_ir_version;
	virtual ~Node(){}
private:
	std::vector<function_parameter> input_params;
	std::vector<function_parameter> output_params;
	// truth table telling if the Nth output is used or not.
	// This might not be as long as the number of outputs in the Node operand's specification
	// (i.e .when trailing outputs are not used)
	std::vector<bool> output_used;

public:
	void set_output_used(std::vector<bool>val){output_used = val; }

	// Get a pointer to the Nth input/output tensor for this node.
	Tensor *get_output_tensor(unsigned N) const;
	Tensor *get_input_tensor(unsigned N) const;
	unsigned get_number_of_inputs(void) const;
	unsigned get_number_of_outputs(void) const;

	// Run caller provided lambda for each output Tensor.
	void forEachOutput( std::function<void(Tensor*)> caller_lambda)
	{
		for( auto op : output_params )
		{
			Tensor* o = std::get<0>(op);
			caller_lambda(o);
		}
	}

	/* Create the C source name. Replace all non a-z,A-Z,0-9 or _
	 * characters. Also prefix name since ONNX allows tensors and nodes
	 * to have the same name */
	std::string c_name(void) const
	{
		return "node_" + cify_name(onnx_name);
	}


	/* Print the C implmementation of the operator */
	virtual void print(std::ostream &destination) const = 0;

	/* Print comma-separated list of function parameters.
	 * Unused optional tensors skipped. e.g.:
	 *   "tensior_X, tensor_Y"
	 * or decorated
	 *   "float tensor_X[1][2][3], float tensor_Y[2][3][4]"
	 *
	 * node::resolve() function creates mappings for all of its function parameters
	 * so that each tensor has a "local name" corresponding to the tensor name in
	 * the ONNX Operands specificaion.
	 */
	void print_parameters(std::ostream &destination, bool decorate ) const;
	void print_function_parameters_definition(std::ostream &destination) const;
	void print_function_parameters_callsite(std::ostream &destination) const;

	/* Figure out in what format the output is in.
	 * This fills the node's list of 'outputs' tensors.
	 * When calling this, the list of 'inputs' must be filled, or the
	 * function fails, segfaults or assumes the input graph is invalid.
	 */
	virtual void resolve(void) {};

	/* Check if an optional output is used in the network.
	 * N is Nth output specified in the Operator.md specification for this node.
	 * Start counting N from 0, including the non-optional outputs. */
	bool is_output_N_used(unsigned N) const;

	/* Replace input tensor 'old' with 'replacement'.
	 * Return false if 'old' is not an input tensor.
	 */
	bool replace_input(Tensor *old, Tensor *replacement);

	/* Not all node types have attributes. Override where needed */
	virtual void parseAttributes( onnx::NodeProto &node )
	{
		ERROR("Attribute parsing not implemented for node operation type " << op_name);
	}

	/* TODO: these should be part of class Tensor... */
	/* Check input constraints, as used in
	 * https://github.com/onnx/onnx/blob/master/docs/Operators.md
	 */
	/* (u)int32, (u)int64, float16/32/64, bfloat*/
	bool typeConstraint_highPrecisionNumeric(const Tensor *t) const;
	/* float16/32/64, bfloat */
	bool typeConstraint_allFloatingPoints(const Tensor *t) const;
	/* float16/32/64, (not bfloat!) */
	bool typeConstraint_plainFloatingPoints(const Tensor *t) const;
	bool typeConstraint_int64(const Tensor *t) const;
	/* int8 or uint8 */
	bool typeConstraint_8bit(const Tensor *t) const;
	/* any integer, signed or not */
	bool typeConstraint_integers(const Tensor *t) const;
	/* only unsigned integers */
	bool typeConstraint_unsigned_integers(const Tensor *t) const;
	/* only signed integers */
	bool typeConstraint_signed_integers(const Tensor *t) const;


	/* Do Multidirectional Broadcasting dimension extensions:
	 * https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
	 */
	void multidirectional_broadcast_size(
		const std::vector<int> A,
		const std::vector<int> B,
		std::vector<int> &result) const;

public:  // TODO: split up into more protected functions
	/* Record a tensor as the generated function's parameter.
	 * - name: the name to be used locally for the tensor in the C-function
	 */
	void register_input(Tensor *, std::string name);
	void register_output(Tensor *, std::string name);
	void name_input(unsigned input_no, std::string name);
	void register_output(unsigned output_no, std::string name);

};
}

