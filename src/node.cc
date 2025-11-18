
#include "error.h"
#include "node.h"


using namespace toC;

int64_t Node::onnx_ir_version;
bool Node::is_output_N_used(unsigned N) const
{
	if( N >= output_used.size() )
		return false;
	return output_used[N];
	return true;
}

bool Node::typeConstraint_highPrecisionNumeric(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
		|| t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_int64(const Tensor *t) const
{
	return (
		t->data_type == onnx::TensorProto_DataType_INT64
	);
}
bool Node::typeConstraint_plainFloatingPoints(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_FLOAT16
		|| t->data_type == onnx::TensorProto_DataType_FLOAT
		|| t->data_type == onnx::TensorProto_DataType_DOUBLE
	);
}
bool Node::typeConstraint_allFloatingPoints(const Tensor *t) const
{
	return (
		   typeConstraint_plainFloatingPoints(t)
		|| t->data_type == onnx::TensorProto_DataType_BFLOAT16
	);
}
bool Node::typeConstraint_8bit(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_INT8
		|| t->data_type == onnx::TensorProto_DataType_UINT8
	);
}

bool Node::typeConstraint_integers(const Tensor *t) const
{
	return (   typeConstraint_unsigned_integers(t)
		|| typeConstraint_signed_integers(t)
	);
}

bool Node::typeConstraint_unsigned_integers(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_UINT8
		|| t->data_type == onnx::TensorProto_DataType_UINT16
		|| t->data_type == onnx::TensorProto_DataType_UINT32
		|| t->data_type == onnx::TensorProto_DataType_UINT64
	);
}
bool Node::typeConstraint_signed_integers(const Tensor *t) const
{
	return (
		   t->data_type == onnx::TensorProto_DataType_INT8
		|| t->data_type == onnx::TensorProto_DataType_INT16
		|| t->data_type == onnx::TensorProto_DataType_INT32
		|| t->data_type == onnx::TensorProto_DataType_INT64
	);
}


void Node::multidirectional_broadcast_size(
	const std::vector<int> A,
	const std::vector<int> B,
	std::vector<int> &result) const
{
		std::vector<int> dim_a = A;
		std::vector<int> dim_b = B;

		while( dim_a.size() < dim_b.size())
			dim_a.insert(dim_a.begin(), 1);
		while( dim_b.size() < dim_a.size())
			dim_b.insert(dim_b.begin(), 1);
		assert(dim_a.size() == dim_b.size());
		for( unsigned i=0; i<dim_a.size(); i++)
		{
			if( dim_a[i] == 1 || dim_b[i] == 1 )
				result.push_back( std::max(dim_a[i], dim_b[i]) );
			else if (dim_a[i] == dim_b[i])
				result.push_back( dim_a[i] );
			else
				ERROR("multidirectional_broadcast: bad tensor shapes for node " << onnx_name);
		}
}



void Node::print_parameters(std::ostream &dst, bool not_callsite ) const
{
	// First create the parameter names as strings (with or without dimensions)
	std::vector<std::string> params;
	for( auto i : input_params ) {
		const Tensor *t = std::get<0>(i);
		std::string name = std::get<1>(i);
		// Unused inputs in the ONNX model still exist in the onnx2c node too
		if( t->is_used() == false )
			continue;
		if( not_callsite )
			params.push_back( t->print_tensor_as_const(name) );
		else
			params.push_back( t->print_tensor_callsite() );
	}
	for( auto o : output_params ) {
		Tensor *t = std::get<0>(o);
		// A node does not know at its resolve time if an optional
		// output is used, so it registers all. Once all nodes
		// are resolved, the tensor knows if some one uses it.
		if( t->is_used() == false )
			continue;
		std::string name = std::get<1>(o);

		// corner case with Shape node: in case the shape output is graph output
		// it is marked const (since other nodes have already used the compile-time generated output
		// of the shape node).
		if( t->isIO )
			t->isConst = false;

		if( not_callsite )
			params.push_back( t->print_tensor(name) );
		else
			params.push_back( t->print_tensor_callsite() );
	}

	// Then print the parmeters as comma-separated string
	auto i = params.begin();
	dst << *i ;
	for( i++; i != params.end(); i++)
		dst << ", " << *i;
}

// parameters at function definition/declaration
void Node::print_function_parameters_definition(std::ostream &destination) const
{
	print_parameters(destination, true);
}
// parameters when calling a function
void Node::print_function_parameters_callsite(std::ostream &destination) const
{
	print_parameters(destination, false);
}

void Node::register_input(Tensor *t, std::string name)
{
	input_params.push_back(function_parameter(t, name));
}
void Node::register_output(Tensor *t, std::string name)
{
	//t->generate=true;
	output_params.push_back(function_parameter(t, name));
}

void Node::name_input(unsigned input_no, std::string name)
{
	std::get<1>(input_params[input_no]) = name;
}
void Node::register_output(unsigned output_no, std::string name)
{
	std::get<1>(output_params[output_no]) = name;
}
Tensor* Node::get_output_tensor(unsigned N) const
{
	if( output_params.size() < N )
		return nullptr;
	return std::get<0>(output_params[N]);
}
Tensor* Node::get_input_tensor(unsigned N) const
{
	if( input_params.size() < N )
		return nullptr;
	return std::get<0>(input_params[N]);
}

unsigned Node::get_number_of_inputs(void) const
{
	return input_params.size();
}

unsigned Node::get_number_of_outputs(void) const
{
	return output_params.size();
}

bool Node::replace_input(Tensor *old, Tensor *replacement)
{

	for( auto &p : input_params )
	{
		if( std::get<0>(p) == old ) {
			LOG(DEBUG) << "Did replacement" << std::endl;
			std::get<0>(p) = replacement;
			return true;
		}
	}

	LOG(DEBUG) << "No replacement" << std::endl;
	return false;
}

std::string Node::math_func(std::string name) const {
	switch (math_type) {
		case onnx::TensorProto_DataType_UNDEFINED:
			ERROR("math function " << name << " called with undefined math type");
			return "";
		case onnx::TensorProto_DataType_FLOAT:
		case onnx::TensorProto_DataType_FLOAT16:
		case onnx::TensorProto_DataType_BFLOAT16:
			return name + "f";
		default:
			return name;
	}
}
