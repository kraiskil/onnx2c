// model resolving part of the toC Graph class
#include "error.h"
#include "graph.h"
#include "onnx.pb.h"

#include "aixlog.hpp"
#include <iostream>

using namespace toC;

int Graph::anonymous_nodes=0;

Graph::Graph(
	onnx::ModelProto &onnx_model,
	bool verbose,
	std::vector<Tensor*> ext_inputs
	)
	:model(onnx_model), verbose_mode(verbose)
{

	AixLog::Severity s = AixLog::Severity::fatal; // there is no "off"
	if( verbose )
		s = AixLog::Severity::trace;
	AixLog::Log::init<AixLog::SinkCerr>(s);


	onnx::GraphProto onnx_graph = onnx_model.graph();

	// 0. add provided external initializers (from test bench
	for( auto t : ext_inputs )
		tensors.push_back(t);

	// 1. add initializers as resolved tensors
	for( auto i : onnx_graph.initializer() )
		addResolvedTensor( i );

	// 2. add graph inputs as resolved tensors
	for ( auto i : onnx_graph.input() ) {
		Tensor *n = getIoTensor( i );
		addTensor( n );
	}
	for ( auto i : onnx_graph.output() ) {
		Tensor *n = getIoTensor( i );
		addTensor( n );
	}

	// while exists unresolved nodes
	//   search in unresolved nodes for a resolvable node (i.e. has resolved inputs)
	//   resolve node + create output tensor
	while( hasUnresolvedNodes() )
	{
		for( auto n : onnx_graph.node() )
			tryResolveNode( n );
	}

}


void Graph::addResolvedTensor(onnx::TensorProto &tensor)
{
	// create Tensor object
	Tensor *t = new Tensor;

	t->parse_onnx_tensor(tensor);

	addTensor(t);
}

Tensor* Graph::getIoTensor(onnx::ValueInfoProto &vi)
{
	onnx::TypeProto tp = vi.type();
	onnx::TypeProto::ValueCase vc = tp.value_case();

	if( vc != onnx::TypeProto::ValueCase::kTensorType )
		ERROR("unimplemented graph input type");

	onnx::TypeProto_Tensor tpt = tp.tensor_type();
	onnx::TensorShapeProto tsp = tpt.shape();

	Tensor *t = new Tensor;
	t->initialize=false;
	t->generate=false;
	t->isIO = true;
	t->name = vi.name();
	t->doc  = vi.doc_string();

	int32_t datatype = tpt.elem_type(); // TODO: check! The onnx.proto doesn't document this explicitly.
	if( onnx::TensorProto_DataType_IsValid(datatype) == false )
		ERROR("Non-valid data type " << datatype << " in tensor " << t->name);
	t->data_type = static_cast<onnx::TensorProto_DataType>(datatype);

	for( onnx::TensorShapeProto_Dimension d : tsp.dim() ) {

		// dim_param is a string that defines this dimension's variable name
		// e.g. "N=1". Used for variable size batches.
		// When the dimension is fixed, the 'param=value' becomes "1=1".
		// For now, all batch sizes are set to 1 in onnx2c generated code

		int dim_size;
		if( isalpha(d.dim_param()[0]) )
			dim_size=1;
		else
			dim_size=d.dim_value();

		t->data_dim.push_back(dim_size);
	}

	return t;
}



bool Graph::nodeInputsResolved(const onnx::NodeProto &node, std::vector<const Tensor*> &inputs)
{
	// if all inputs can be found in the tensors-vector, then yes, inputs are resolved
	for( auto i : node.input() )
	{
		bool input_resolved = false;
		for( auto t : tensors ) {
			if ( t->name == i ) {
				input_resolved = true;
				inputs.push_back(t);
				break;
			}
		}

		// Node has an unresolved input tensor
		if( input_resolved == false )
			return false;
	}

	return true;
}

void Graph::tryResolveNode(onnx::NodeProto &node)
{
	// Check if node has all inputs resolved
	std::vector<const Tensor*> inputs;
	std::vector<Tensor*> outputs;


	if( nodeInputsResolved(node, inputs) == false )
		return;


	Node *n = findNode(node.op_type());
	n->isResolved = false;
	n->op_name = node.op_type();
	n->onnx_name = node.name();

	if( n->onnx_name == "" ) {
		std::string name = "anonymous_";
		name += n->op_name;
		name +=  "_" + std::to_string(anonymous_nodes);
		n->onnx_name = name;
	}

	if( node.attribute_size() != 0 )
		n->parseAttributes( node );

	n->resolveOutput(inputs, outputs );


	// Loop over node's declared outputs.
	// This will now contain all of the node's outputs, also such optional ones
	// that are not used in the model.
	for( unsigned o=0; o<outputs.size(); o++) {
		Tensor *t = outputs[o];

		// optional outputs are named "" or just omitted
		std::string onnx_name="";
		if( (int)o<node.output_size() )
			onnx_name = node.output(o);

		if( onnx_name == "" ) {
			if (t->isRecursive==false) {
				LOG(TRACE) << "skipping: output number " << o << " is unused" << std::endl;
				continue;
			}
			onnx_name = n->c_name() + "_recursive_"+std::to_string(o);
		}
		t->name = onnx_name;

		addTensor(t);
	}

	n->isResolved = true;
	nodes.push_back(n);
	LOG(DEBUG) << "Adding " << n->op_name << " node: " << n->onnx_name << std::endl;
	LOG(DEBUG) << "    inputs: " << std::endl;
	for( auto i : inputs)
		LOG(DEBUG) << "         " << i->name << std::endl;
	LOG(DEBUG) << "    outputs: " << n->onnx_name << std::endl;
	for( auto o : outputs)
		LOG(DEBUG) << "         " << o->name << std::endl;
}


bool Graph::hasUnresolvedNodes(void)
{
	return model.graph().node_size() > (int)nodes.size();
}


#include "nodes/add.h"
#include "nodes/batchnormalization.h"
#include "nodes/conv.h"
#include "nodes/flatten.h"
#include "nodes/lstm.h"
#include "nodes/matmul.h"
#include "nodes/maxpool.h"
#include "nodes/relu.h"
#include "nodes/reshape.h"
#include "nodes/sigmoid.h"
#include "nodes/squeeze.h"
#include "nodes/softmax.h"
#include "nodes/transpose.h"

Node* Graph::findNode(std::string opName)
{
	if( opName == "Add" )return new Add;
	if( opName == "BatchNormalization" )return new BatchNormalization;
	if( opName == "Conv" )return new Conv;
	if( opName == "Flatten" )return new Flatten;
	if( opName == "LSTM" )return new LSTM;
	if( opName == "MatMul" )return new MatMul;
	if( opName == "MaxPool" )return new MaxPool;
	if( opName == "Relu" )return new Relu;
	if( opName == "Reshape" )return new Reshape;
	if( opName == "Sigmoid" )return new Sigmoid;
	if( opName == "Squeeze" )return new Squeeze;
	if( opName == "Softmax" )return new Softmax;
	if( opName == "Transpose" )return new Transpose;

	ERROR("Unimplemented: node operation " << opName);
	return NULL;
}

void Graph::addTensor(Tensor *t)
{
	/* This is a bit fragile!
	 * Add tensor t to known tensors, or if
	 * a tensor of the same name already exists,
	 * update the existing tensor.
	 *
	 * How and which parts of the existing tensor
	 * to update is the fragile part :)
	 */
	Tensor *prev = NULL;
	for( auto o : tensors)
		if( t->name == o->name ) {
			prev = o;
			break;
		}

	if( prev == NULL ) {
		tensors.push_back(t);
		LOG(DEBUG) << "Adding new tensor: " << t->name << " - "<< t->data_type_str() << " { " << t->str_dimensions() << "}" << std::endl;
	}
	else {
		LOG(DEBUG) << "Updating existing tensor: " << t->name << std::endl;
		LOG(TRACE) << "   was: gen " << prev->generate
		           << "  init " << prev->initialize
		           << "  IO " << prev->isIO
		           << "  recurs " << prev->isRecursive
		           << "  alias " << !!prev->isAliasOf
		           << std::endl;
		LOG(TRACE) << "   new: gen " << t->generate
		           << "  init " << t->initialize
		           << "  IO " << t->isIO
		           << "  recurs " << t->isRecursive
		           << "  alias " << !!t->isAliasOf
		           << std::endl;


		// if updating an output to be recursive:
		if( t->isRecursive ) {
			// Since this tensor was already added, it was added
			// because it is a graph output.
			// This is because recursion means recursion to same node, not a general loop in the network
			if( prev->isIO == false )
				ERROR("Update logic failure (i.e. this is an assert fail)");
			if( t->isAliasOf ) {
				prev->generate = false;
				prev->initialize = false;
				prev->isAliasOf = t->isAliasOf;
			}
			else {
				prev->generate = t->generate;
				prev->initialize = t->initialize;
			}
			prev->isRecursive = true;
		}

		LOG(TRACE) << "   now: gen " << prev->generate
		           << "  init " << prev->initialize
		           << "  IO " << prev->isIO
		           << "  recurs " << prev->isRecursive
		           << "  alias " << !!prev->isAliasOf
		           << std::endl;
	}
}

Tensor *Graph::findTensor(const std::string &name) const
{
	for( auto o : tensors)
		if( o->name == name )
			return o;
	return NULL;
}

