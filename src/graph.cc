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
		s = AixLog::Severity::debug;
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
	t->generate=false;
	t->initialize=false;
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
	n->inputs = inputs;

	if( n->onnx_name == "" ) {
		std::string name = "anonymous_";
		name += n->op_name;
		name +=  "_" + std::to_string(anonymous_nodes);
		n->onnx_name = name;
	}

	if( node.attribute_size() != 0 )
		n->parseAttributes( node );

	n->resolveOutput(inputs, outputs );


	for( unsigned o=0; o<outputs.size(); o++) {
		Tensor *t = outputs[o];

		if ( (int)o >= node.output_size() ) {
			LOG(TRACE) << "skipping optional output after number " << o << std::endl;
			break;
		}
		t->name = node.output(o);
		t->generate=true;
		t->initialize=false;
		addTensor(t);
		n->outputs.push_back(t);
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
#include "nodes/matmul.h"
#include "nodes/maxpool.h"
#include "nodes/relu.h"
#include "nodes/reshape.h"
#include "nodes/sigmoid.h"
#include "nodes/softmax.h"
#include "nodes/transpose.h"

Node* Graph::findNode(std::string opName)
{
	if( opName == "Add" )return new Add;
	if( opName == "BatchNormalization" )return new BatchNormalization;
	if( opName == "Conv" )return new Conv;
	if( opName == "Flatten" )return new Flatten;
	if( opName == "MatMul" )return new MatMul;
	if( opName == "MaxPool" )return new MaxPool;
	if( opName == "Relu" )return new Relu;
	if( opName == "Reshape" )return new Reshape;
	if( opName == "Sigmoid" )return new Sigmoid;
	if( opName == "Softmax" )return new Softmax;
	if( opName == "Transpose" )return new Transpose;

	ERROR("Unimplemented: node operation " << opName);
	return NULL;
}

bool Graph::addTensor(Tensor *t)
{
	/* ONNX allows (but does not require - or there are bugs out there)
	 * for initializer tensors to be listed as inputs. Those have been
	 * processed elsewhere already. */
	bool pushit = true;
	for( auto o : tensors)
		if( t->name == o->name )
			pushit = false;

	if( pushit ) {
		tensors.push_back(t);
		LOG(DEBUG) << "Adding tensor: " << t->name << " - "<< t->data_type_str() << " { " << t->str_dimensions() << "}" << std::endl;

	}
	return pushit;
}

Tensor *Graph::findTensor(const std::string &name) const
{
	for( auto o : tensors)
		if( o->name == name )
			return o;
	return NULL;
}

