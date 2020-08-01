// model resolving part of the toC Graph class
#include "error.h"
#include "graph.h"
#include "onnx.pb.h"

#include <iostream>

using namespace toC;

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

	t->generate=true;
	t->initialize=true;

	// assert tensor is resolvable
	if( onnx::TensorProto_DataLocation() != onnx::TensorProto_DataLocation_DEFAULT )
		ERROR("unhandled: non-default data location in tensor " << tensor.name());
	if( tensor.has_segment() )
		ERROR("unhandled: segmented data in tensor" << tensor.name());

	// populate tensor
	int data_dimensions = tensor.dims_size();
	if( data_dimensions < 1 || data_dimensions > tensor_max_dim )
		ERROR("unhandled number of dimensions: " << data_dimensions);
	for( auto &i : t->data_dim )
		i=0;
	for( int i=0; i<data_dimensions; i++)
		t->data_dim[i] = tensor.dims(i);

	int32_t datatype = tensor.data_type();
	if( onnx::TensorProto_DataType_IsValid(datatype) == false )
		ERROR("Non-valid data type " << datatype << " in tensor " << tensor.name());
	t->data_type = static_cast<onnx::TensorProto_DataType>(datatype);

	int64_t data_num_elem; // Can data size be negative? onnx.pb.h encodes size into 'signed int'
	switch( datatype )
	{
		case onnx::TensorProto_DataType_FLOAT:
			data_num_elem = tensor.float_data_size(); break;

		case onnx::TensorProto_DataType_INT32:
			data_num_elem = tensor.int32_data_size(); break;
		default:
			ERROR("unhandled tensor data type in tensor " << tensor.name());
			break;
	};
	t->data_num_elem = data_num_elem;
	t->data_buffer = malloc(data_num_elem * t->data_elem_size());
	if( t->data_buffer == NULL )
		ERROR("memory allocation failed for tensor " << tensor.name());
	switch( datatype )
	{
		case onnx::TensorProto_DataType_FLOAT:
			for( int i=0; i<data_num_elem; i++  )
				((float*)t->data_buffer)[i] = tensor.float_data(i);
			break;
		case onnx::TensorProto_DataType_INT32:
			for( int i=0; i<data_num_elem; i++  )
				((int32_t*)t->data_buffer)[i] = tensor.int32_data(i);
			break;
		default:
			ERROR("unhandled tensor data type in tensor " << tensor.name());
			break;
	};

	t->name = tensor.name();
	t->doc = tensor.doc_string();

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

	for( onnx::TensorShapeProto_Dimension d : tsp.dim() ) {

		// TODO: too much fail in the next 5 lines to warrant any better comment!
		int dim_param;
		if( isalpha(d.dim_param()[0]) )
			dim_param=1;
		else
			dim_param=atoi(d.dim_param().c_str());

		if( d.has_dim_param() == false )
			dim_param=1;
		if( d.dim_value() >= tensor_max_dim )
			ERROR("Unimplmeneted - input of too many dimensions");

		t->data_dim[d.dim_value()] = dim_param;
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

	if( node.attribute_size() != 0 )
		ERROR("unhandled: node attributes in " << node.name() );

	if( nodeInputsResolved(node, inputs) == false )
		return;


	Node *n = new Node;
	n->isResolved = false;
	n->name = node.name();
	n->op_name = node.op_type();
	n->inputs = inputs;



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
