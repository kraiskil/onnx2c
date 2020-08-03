// model resolving part of the toC Graph class
#include "error.h"
#include "graph.h"
#include "onnx.pb.h"

#include <iostream>

using namespace toC;

int Graph::anonymous_nodes=0;

Graph::Graph(onnx::ModelProto &onnx_model)
	:model(onnx_model)
{
	initializeOpArray();

	onnx::GraphProto onnx_graph = onnx_model.graph();

	// 1. add initializers as resolved tensors
	for( auto i : onnx_graph.initializer() )
		addResolvedTensor( i );

	// 2. add graph inputs as resolved tensors
	for ( auto i : onnx_graph.input() )
		tensors.push_back(getIoTensor( i ));

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

	// add Tensor to database
	tensors.push_back(t);
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
	t->name = vi.name();
	t->doc  = vi.doc_string();

	int32_t datatype = tpt.elem_type(); // TODO: check! The onnx.proto doesn't document this explicitly.
	if( onnx::TensorProto_DataType_IsValid(datatype) == false )
		ERROR("Non-valid data type " << datatype << " in tensor " << t->name);
	t->data_type = static_cast<onnx::TensorProto_DataType>(datatype);

	int64_t num_elem=1;
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

		num_elem *=dim_size;
		t->data_dim.push_back(dim_size);
	}

	t->data_num_elem = num_elem;
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

	if( node.attribute_size() != 0 )
		ERROR("unhandled: node attributes in " << node.name() );

	if( nodeInputsResolved(node, inputs) == false )
		return;


	Node *n = new Node;
	n->isResolved = false;
	n->op_name = node.op_type();
	n->inputs = inputs;
	n->name = node.name();

	if( n->name == "" ) {
		std::string name = "anonymous_";
		name += n->op_name;
		name +=  "_" + std::to_string(anonymous_nodes);
		n->name = name;
	}


	n->op = findOp(n->op_name);
	n->op->resolveOutput(inputs, outputs );

	// TODO: looking at onnx.proto, seems a node can have multiple outputs
	//       but how to map output names to the outputs?
	if( outputs.size() != 1 )
		ERROR("Unimplemented - more (or less) than one output");
	assert(node.output_size() == 1);

	Tensor *t = outputs[0];
	t->name = node.output(0); 
	t->generate=true;
	t->initialize=false;
	tensors.push_back(t);

	for( auto o : outputs)
		n->outputs.push_back(o);

	n->isResolved = true;
	nodes.push_back(n);
}


bool Graph::hasUnresolvedNodes(void)
{
	return model.graph().node_size() > (int)nodes.size();
}


const Op* Graph::findOp(std::string opName)
{
	for( auto o: ops )
		if( o->name == opName )
			return o;

	ERROR("Unimplemented: node operation " << opName);
	return NULL;
}
