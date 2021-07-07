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
	Node::onnx_ir_version = onnx_ir_version();
	// 0. add provided external initializers (from test bench
	for( auto t : ext_inputs )
		tensors.push_back(t);

	// 1. add initializers as resolved tensors
	// in case of quantization, make quantized copies here
	for( auto i : onnx_graph.initializer() )
		addInitializedTensor( i );

	// 2. add graph inputs as resolved tensors
	// in case of quantization, convert all IO to INT8
	for ( auto i : onnx_graph.input() ) {
		Tensor *n = getIoTensor( i );
		addTensor( n );
	}
	for ( auto i : onnx_graph.output() ) {
		Tensor *n = getIoTensor( i );
		addTensor( n );
	}

	// Resolve all nodes.
	// TODO: this now walks through all nodes in the order
	// they are defined in the onnx file. This works only
	// because all input graphs encountered thus far have
	// been resolvable in that order.
	// But there is no guarantee (that I found) in onnx
	// that says nodes could not be given in other orders too.
	// Fix this when (if) someone can produce such a file
	for( auto n : onnx_graph.node() )
		tryResolveNode( n );
}


/* Add already resolved onnx::TensorProto. E.g. TensorProtos that
 * are resolved already in the ONNX model (inputs and initialized ones)
 */
void Graph::addInitializedTensor(onnx::TensorProto &tensor)
{
	Tensor *t = new Tensor;

	t->parse_onnx_tensor(tensor);
	t->isConst = true;

	addTensor(t);

	if( quantize ) {
		t = t->make_quantized_copy();
		if( t )
			addTensor(t);
	}
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
	t->isConst = false;
	t->name = vi.name();
	t->doc  = vi.doc_string();

	int32_t datatype = tpt.elem_type(); // TODO: check! The onnx.proto doesn't document this explicitly.
	if( onnx::TensorProto_DataType_IsValid(datatype) == false )
		ERROR("Non-valid data type " << datatype << " in tensor " << t->name);
	t->data_type = static_cast<onnx::TensorProto_DataType>(datatype);

	// TODO: this is a bit coarse
	if( quantize )
		t->data_type = onnx::TensorProto_DataType_INT8;

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



bool Graph::getNodeInputTensors(const onnx::NodeProto &node, std::vector<const Tensor*> &inputs)
{
	// TODO: ugly. Move where?
	static const Tensor unused;

	// if all inputs can be found in the tensors-vector, then yes, inputs are resolved
	for( auto i : node.input() )
	{
		bool input_resolved = false;
		// Unused inputs don't need to be resolved.
		if( i == "" ) {
			input_resolved = true;
			inputs.push_back(&unused);
			continue;
		}

		for( auto t : tensors ) {
			if ( t->name == i ) {
				input_resolved = true;
				inputs.push_back(t);
				break;
			}
		}

		// Node has an unresolved input tensor
		if( input_resolved == false ) {
			LOG(TRACE) << "Input tensor " << i << " not resolved" << std::endl;
			return false;
		}
	}

	return true;
}

void Graph::tryResolveNode(onnx::NodeProto &node)
{
	std::vector<const Tensor*> inputs;
	std::vector<Tensor*> outputs;
	LOG(DEBUG) << "Resolving node " << node.name() <<std::endl;

	// Early exit on error cases - cannot resolve this node (now)
	for( auto o : nodes )
		if( node.name() == o->onnx_name ) {
			LOG(TRACE) << "Node " << node.name() << " already resolved"<<std::endl;
			return;
		}
	if( getNodeInputTensors(node, inputs) == false )
		return;


	// ONNX has a few nodes that have quantized alternatives.
	// Switch to those here.
	// For the rest, rely on optional quantization in the
	// onnx2c implementation.
	std::string new_node = node.op_type();
	if( quantize ) {
		replaceWithQuantized(inputs);
		if( new_node == "Conv" )
			new_node = "ConvInteger";
		if( new_node == "MatMul" )
			new_node = "MatMulInteger";
	}
	Node *n = createNode(new_node);

	n->onnx_node = &node;
	n->isResolved = false;
	n->op_name = new_node;
	n->onnx_name = node.name();

	// onnx allows (or at least some tools create) nodes without names
	// create unique names for those, e.g. "anonymous_5_relu"
	if( n->onnx_name == "" ) {
		std::string name = "anonymous_";
		name += n->op_name;
		name +=  "_" + std::to_string(anonymous_nodes);
		n->onnx_name = name;
		anonymous_nodes++;
	}

	if( node.attribute_size() != 0 )
		n->parseAttributes( node );

	// Bad name - see issue #5. "resolveNode()", maybe?
	n->resolveOutput(inputs, outputs );


	// Loop over node's declared outputs.
	// This will now contain all of the node's outputs, also such optional ones
	// that are not used in the model.
	for( unsigned o=0; o<outputs.size(); o++) {
		Tensor *t = outputs[o];

		// optional outputs are named "" or just omitted
		std::string onnx_name;
		if( n->is_output_N_used(o) )
			onnx_name = node.output(o);
		else
			onnx_name = "";

		// recursive nodes are special: if they are not used by other nodes,
		// then the ONNX graph doesn't record them (i.e. they look like they'd be unused)
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

// Turns out there are two versions of ONNX IR
// the "format" and the "OperatorSetId".
// What is interesting here is the versions of the operators
int64_t Graph::onnx_ir_version(void)
{
	int opset_import_size = model.opset_import_size();
	if( opset_import_size != 1 )
		ERROR("Model has multiple opset versions. This is legal, but really needs better documentation on what to do now.");
	auto foo = model.opset_import(0);
	int64_t version = foo.version();
	return version;
}

#include "nodes/arithmetic.h"
#include "nodes/averagepool.h"
#include "nodes/batchnormalization.h"
#include "nodes/concat.h"
#include "nodes/constant.h"
#include "nodes/conv.h"
#include "nodes/convinteger.h"
#include "nodes/dropout.h"
#include "nodes/dynamicquantizelinear.h"
#include "nodes/elementwise.h"
#include "nodes/flatten.h"
#include "nodes/gemm.h"
#include "nodes/globalaveragepool.h"
#include "nodes/lstm.h"
#include "nodes/matmul.h"
#include "nodes/matmulinteger.h"
#include "nodes/maxpool.h"
#include "nodes/relu.h"
#include "nodes/reshape.h"
#include "nodes/sigmoid.h"
#include "nodes/squeeze.h"
#include "nodes/softmax.h"
#include "nodes/transpose.h"
#include "nodes/unsqueeze.h"

Node* Graph::createNode(std::string opName)
{
	if( opName == "Abs" )return new Elementwise("Abs");
	if( opName == "Add" )return new Arithmetic("Add");
	if( opName == "AveragePool" )return new AveragePool;
	if( opName == "BatchNormalization" )return new BatchNormalization;
	if( opName == "Ceil" )return new Elementwise("Ceil");
	if( opName == "Concat" )return new Concat;
	if( opName == "Constant" )return new Constant;
	if( opName == "Conv" )return new Conv;
	if( opName == "ConvInteger" )return new ConvInteger;
	if( opName == "Div" )return new Arithmetic("Div");
	if( opName == "Dropout" )return new Dropout;
	if( opName == "DynamicQuantizeLinear" )return new DynamicQuantizeLinear;
	if( opName == "Flatten" )return new Flatten;
	if( opName == "Floor" )return new Elementwise("Floor");
	if( opName == "GlobalAveragePool" )return new GlobalAveragePool;
	if( opName == "Gemm" )return new Gemm;
	if( opName == "LSTM" )return new LSTM;
	if( opName == "MatMul" )return new MatMul;
	if( opName == "MatMulInteger" )return new MatMulInteger;
	if( opName == "MaxPool" )return new MaxPool;
	if( opName == "Mul" )return new Arithmetic("Mul");
	if( opName == "Relu" )return new Relu;
	if( opName == "Reshape" )return new Reshape;
	if( opName == "Sigmoid" )return new Sigmoid;
	if( opName == "Squeeze" )return new Squeeze;
	if( opName == "Softmax" )return new Softmax;
	if( opName == "Transpose" )return new Transpose;
	if( opName == "Unsqueeze" )return new Unsqueeze;

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
		// TODO return & remove else {}
	}
	else {
		LOG(DEBUG) << "Updating existing tensor: " << t->name << std::endl;
		LOG(TRACE) << "   was: gen " << prev->generate
		           << "  init " << prev->initialize
		           << "  IO " << prev->isIO
		           << "  const " << prev->isConst
		           << "  recurs " << prev->isRecursive
		           << std::endl;
		LOG(TRACE) << "   new: gen " << t->generate
		           << "  init " << t->initialize
		           << "  IO " << t->isIO
		           << "  const " << prev->isConst
		           << "  recurs " << t->isRecursive
		           << std::endl;


		// if updating an output to be recursive:
		if( t->isRecursive ) {
			// Since this tensor was already added, it was added
			// because it is a graph output.
			// This is because recursion means recursion to same node, not a general loop in the network
			if( prev->isIO == false )
				ERROR("Update logic failure (i.e. this is an assert fail)");
			/*
			if( t->isAliasOf ) {
				prev->generate = false;
				prev->initialize = false;
				prev->isAliasOf = t->isAliasOf;
			}
			else {
			*/
				prev->generate = t->generate;
				prev->initialize = t->initialize;
			//}
			prev->isRecursive = true;
		}
		// Recursive nodes might need to initialize internal tensors
		if( t->data_buffer )
			prev->data_buffer = t->data_buffer;

		if( t->isConst == false )
			prev->isConst = false;

		prev->initialize = t->initialize || prev->initialize;
		if( prev->initialize )
			prev->generate=true;

		LOG(TRACE) << "   now: gen " << prev->generate
		           << "  init " << prev->initialize
		           << "  IO " << prev->isIO
		           << "  const " << prev->isConst
		           << "  recurs " << prev->isRecursive
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

void Graph::replaceWithQuantized(std::vector<const Tensor*> &inputs)
{
	for( unsigned i=0; i<inputs.size(); i++ ) {
		if(inputs[i]->quantizedCopy)
			inputs[i] = inputs[i]->quantizedCopy;
	}
}

