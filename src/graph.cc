// model resolving part of the toC Graph class
#include "error.h"
#include "graph.h"
#include "nodes/graph_io.h"
#include "onnx.pb.h"
#include "options.h"

#include "aixlog.hpp"
#include <iostream>


using namespace toC;

int Graph::anonymous_nodes=0;


Graph::Graph(
	onnx::ModelProto &onnx_model,
	std::vector<Tensor*> ext_inputs
	)
	:model(onnx_model)
{

	processGraph(onnx_model, ext_inputs);
}

void Graph::processGraph(
	onnx::ModelProto &onnx_model,
	std::vector<Tensor*> ext_inputs
	)
{
	onnx::GraphProto onnx_graph = onnx_model.graph();
	Node::onnx_ir_version = onnx_ir_version();
	// 0. add provided external initializers (from test bench
	LOG(DEBUG) << "Adding external (testsuite) tensors." <<std::endl;
	for( auto t : ext_inputs ) {
		LOG(DEBUG) << "  - " << t->name <<std::endl;
		tensors.push_back(t);
	}
	LOG(TRACE) << "  (done adding external tensors)." <<std::endl;

	// 1. add initializers as resolved tensors
	LOG(DEBUG) << "Adding initialized constant tensors from .onnx file." <<std::endl;
	for( auto i : onnx_graph.initializer() )
		addInitializedTensor( i );
	LOG(TRACE) << "  (done adding initialized tensors)." <<std::endl;

	// 2. add graph inputs as resolved tensors
	// in case of quantization, convert all IO to INT8
	addGraphInputMetanode();
	LOG(DEBUG) << "Marking graph input tensors as IO." <<std::endl;
	for ( auto i : onnx_graph.input() ) {
		// NB: onnx:Graph:input() gives graph inputs AND initialized tensors
		// filter out the initizlized (i.e. const) tensors.
		// They are not graph inputs in the sense of onnx2c::Tensor::isIO
		if( findTensorByName( i.name()) == nullptr ) {
			Tensor *n = getIoTensor( i );
			addTensor( n );
		}
	}
	LOG(TRACE) << "  (done marking input tensors)." <<std::endl;

	// 3. Do the nodes
	LOG(DEBUG) << "Resolving nodes." <<std::endl;
	resolveGraphNodes(onnx_graph);

	// 4. Add the IO tag to those tensors the user wants back.
	Node *graph_output_node = addGraphOutputMetanode();
	LOG(DEBUG) << "Marking graph output tensors as IO." <<std::endl;
	for ( auto o : onnx_graph.output() ) {
		LOG(TRACE) << "\t- found graph output tensor '" << o.name() << "':" << std::endl;
		Tensor *t = findTensor(o.name());
		if( t == nullptr )
			ERROR("Badly formed ONNX graph: No node produced this graph output tensor");
		t->isIO = true;
		// There is the odd case (in tests, mostly), where an constant tensor is passed
		// as graph output. Only in this case should the graph's output be generated into
		// the C source
		t->generate = t->isConst;

		t->consumers.push_back(graph_output_node);
		graph_output_node->register_input(t, "");
		LOG(TRACE) << "\t\t " << t->print_trace_dump() << std::endl;
	}
}

void Graph::resolveGraphNodes(onnx::GraphProto &onnx_graph)
{
	/* A vast majority of ONNX graphs in the wild list their
	 * nodes in an order where they can be resolved in the
	 * order they are listed in the onnx file.
	 * This "brute force" search for resolvable nodes should
	 * not affect speed at all in the normal case
	 * "Optimize for speed" here that case. If the input graph
	 * is big enough that this algorithm becomes a bottleneck,
	 * then onnx2c probably isn't your best option anyway.
	 */

	unsigned num_unresolved_prev_round;
	unsigned num_unresolved = onnx_graph.node_size();

	do {
		num_unresolved_prev_round = num_unresolved;
		num_unresolved = 0;

		for( onnx::NodeProto n : onnx_graph.node() ) {
			bool res = tryResolveNode( n );
			if( res == false )
				num_unresolved++;
		}

		// all done
		if( num_unresolved == 0 )
			break;

	// repeat as long as at least one new node got resolved
	} while ( num_unresolved < num_unresolved_prev_round );

	if( num_unresolved != 0 )
		ERROR("Input ONNX graph is not resolvable.");
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

	for( onnx::TensorShapeProto_Dimension d : tsp.dim() ) {

		// dim_param is a string that defines this dimension's variable name
		// e.g. "N=1" or "batch_size". Seems to be used for variable size batches.
		// When the dimension is fixed, the variable does not have a name, and param
		// is the string representation of value (i.e. "1=1").
		// Onnx2c doesn't allow variable batch sizes, so if no value is given, use 1.
		int dim_size;
		if( isalpha(d.dim_param()[0]) ) {
			if( d.dim_value() ) {
				dim_size=d.dim_value();
			}
			else {
				uint32_t user_value = options.dim_defines[d.dim_param()];
				if( user_value == 0 ) {
					LOG(WARNING) << "Graph input tensor dimension (" << d.dim_param() << ") not specified!" << std::endl;
					LOG(WARNING) << "Specify with command line option '-d "<< d.dim_param() <<":<value>'" << std::endl;
					LOG(WARNING) << "Defining this dimension as 1 for now."<< std::endl;
					dim_size=1;
				}
				else {
					LOG(DEBUG) << "Graph input tensor dimension (" << d.dim_param() << ") set on command line to " << user_value << std::endl;
					dim_size = user_value;
				}
			}
		}
		else {
			dim_size=d.dim_value();
		}

		t->data_dim.push_back(dim_size);
	}

	return t;
}


// Populate the onnx2c_node's
// input tensors using already created tensors in the graph.
// All inputs should exist as onnx2c tensors before calling this, or else we return false.
bool Graph::getNodeInputTensors(const onnx::NodeProto &node, toC::Node *onnx2c_node)
{
	// Step through the ONNX node's input tensors
	for( auto i : node.input() )
	{
		bool input_resolved = false;
		// in case the input is not used by the node, ONNX has a dummy input
		// for the node. This dummy input serves only to put the rest of the
		// node's inputs in correct order
		if( i == "" ) {
			static Tensor input_is_unused_sentinel;
			LOG(TRACE) << "\t-unnamed input tensor - using shared 'unused' sentinel tensor" << std::endl;
			input_resolved = true;
			onnx2c_node->register_input(&input_is_unused_sentinel, "");
			continue;
		}

		LOG(TRACE) << "Looking for input tensor '" << i << "':" << std::endl;
		for( auto t : tensors ) {
			if ( t->name == i ) {
				LOG(TRACE) << "\t- found input tensor '" << i << "':" << std::endl;
				LOG(TRACE) << "\t\t " << t->print_trace_dump() << std::endl;
				input_resolved = true;
				// register node with local name "" - since we don't have node context here
				// we don't know if it is named 'X', 'input', 'A' or whatever. Node resolver
				// assigns that name.
				onnx2c_node->register_input(t, "");
				break;
			}
		}
		LOG(TRACE) << "    finished looking" << std::endl;

		// Node has an unresolved input tensor
		if( input_resolved == false ) {
			LOG(DEBUG) << "Input tensor '" << i << "' not resolved" << std::endl;
			return false;
		}
	}

	return true;
}


/* Make an onnx2c node object out of the onnx::NodeProto object.
 * @return true node is (or was earlier) added to Graph::nodes datastructure.
 *          Return false if Graph::tensors does not yet have all the input tensor for this node.
 */
bool Graph::tryResolveNode(onnx::NodeProto &onnx_node)
{
	LOG(DEBUG) << "Resolving ONNX node: '" << onnx_node.name() << "'" <<std::endl;

	// This check is needed in case the caller needs to iterate over the nodes more than once.
	for( auto o : nodes )
		if( onnx_node.name() == o->onnx_name ) {
			LOG(TRACE) << "Node '" << onnx_node.name() << "' already resolved"<<std::endl;
			return true;
		}

	Node *n = createNode(onnx_node);

	// TODO: add comment explaining what is going on here...
	if( getNodeInputTensors(onnx_node, n) == false ) {
		LOG(TRACE) << "getNodeInputTensors() failed. Not adding node!"<< std::endl;
		delete n;
		return false;
	}

	// ONNX allows (or at least some tools create) nodes without names.
	// Here we create unique names for those, e.g. "anonymous_5_relu".
	if( onnx_node.name() == "" ) {
		std::string name = "anonymous_";
		name += n->op_name;
		name +=  "_" + std::to_string(anonymous_nodes);
		n->onnx_name = name;
		anonymous_nodes++;
	}
	else
		n->onnx_name = onnx_node.name();

	LOG(TRACE) << "    Node name in C sources " << n->c_name() << std::endl;
	LOG(TRACE) << "    inputs: " << std::endl;

	// Record this node as the consumer of the the input tensors
	for(unsigned iidx=0; iidx<(n->get_number_of_inputs()); iidx++) {
		Tensor *i = n->get_input_tensor(iidx);
		LOG(TRACE) << "         " << i->name << " - "<< i->data_type_str() << " { " << i->str_dimensions() << "}" << std::endl;
		const_cast<Tensor*>(i)->consumers.push_back(n);
		i->print_trace_dump();
	}
	LOG(TRACE) << "     (no more inputs)" << std::endl;
	n->isResolved = false;

	LOG(TRACE) << "  Parsing node attributes" << std::endl;
	if( onnx_node.attribute_size() != 0 )
		n->parseAttributes( onnx_node );
	LOG(TRACE) << "    (done parsing attributes)" << std::endl;

	// Now loop over the node inputs, check that they are all added
	// into the graph's known tensors - seems the ONNX graph does not keep track of
	// vectors provided as nodes' attributes.
	// TODO: how does this work? attribute tensors are added as node inputs?
	//       Are they a part of the arguments to a function call in C?
	LOG(TRACE) << "  Making sure node attributes are in the graph" << std::endl;
	for(unsigned nn = 0; nn<n->get_number_of_inputs(); nn++)
		addTensor(n->get_input_tensor(nn));
	LOG(TRACE) << "   (end of attribute-input-vectors)" << std::endl;

	// create output nodes for the tensor.
	// this is a kludge around a chicken & egg problem caused by bad design in
	// onnx2c:
	// we don't want to save a copy of the onnx::NodeProto in the onnx2c::node object
	// (since who knows how protobuf keeps its internals).
	// So create a list of that tells if outputs are used or not *before* resolving
	// the node.
	std::vector<bool> output_used;
	for(int nn = 0; nn<onnx_node.output_size(); nn++)
	{
		// ONNX spec:
		// "There are two ways to leave an optional input or output unspecified:
		// the first, available only for trailing inputs and outputs, is to simply
		// not provide that input; the second method is to use an empty string in
		// place of an input or output name."
		if( onnx_node.output(nn) == "" )
			output_used.push_back(false);
		else
			output_used.push_back(true);
	}
	n->set_output_used(output_used);

	// Configure Node internals, and populate its outputs vector.
	LOG(TRACE) << "Resolving node" << std::endl;
	n->resolve();

	// Add the output tensors the resolve() generated to the graph's list of tensors.
	// Name the generated output tensors according to how they are named in
	// the ONNX model.
	// This will now contain all of the node's outputs, also such optional ones
	// that are not used in the model.
	LOG(TRACE) << "Adding resolved node's output to graph's tensors" << std::endl;
	for( unsigned o=0; o<n->get_number_of_outputs(); o++) {
		Tensor *t = n->get_output_tensor(o);

		// optional outputs are named "" or just omitted
		std::string onnx_name;
		if( n->is_output_N_used(o) )
			onnx_name = onnx_node.output(o);
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
	LOG(TRACE) << "   (done) all outputs now:" << std::endl;
	for( unsigned o=0; o<n->get_number_of_outputs(); o++) {
		Tensor *t = n->get_output_tensor(o);
		LOG(TRACE) << "         " << t->name << " - "<< t->data_type_str() << " { " << t->str_dimensions() << "}" << std::endl;
	}
	LOG(TRACE) << "      (no more outputs)" << std::endl;

	log_trace_all_tensors();
	n->isResolved = true;
	nodes.push_back(n);
	return true;
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
	if( opset_import_size == 0 )
		ERROR("Model has no opset version");
	if( opset_import_size > 1 )
		LOG(INFO) << "Model has multiple opset versions." << std::endl;
	auto foo = model.opset_import(0);
	int64_t version = foo.version();
	return version;
}

#include "nodes/averagepool.h"
#include "nodes/batchnormalization.h"
#include "nodes/cast.h"
#include "nodes/clip.h"
#include "nodes/concat.h"
#include "nodes/constant.h"
#include "nodes/constantofshape.h"
#include "nodes/conv.h"
#include "nodes/convinteger.h"
#include "nodes/convtranspose.h"
#include "nodes/dequantizelinear.h"
#include "nodes/dropout.h"
#include "nodes/dynamicquantizelinear.h"
#include "nodes/elementwise.h"
#include "nodes/elementwise_2.h"
#include "nodes/elementwise_variadic.h"
#include "nodes/expand.h"
#include "nodes/flatten.h"
#include "nodes/gather.h"
#include "nodes/gemm.h"
#include "nodes/globalaveragepool.h"
#include "nodes/globalmaxpool.h"
#include "nodes/identity.h"
#include "nodes/instancenorm.h"
#include "nodes/layernorm.h"
#include "nodes/lrn.h"
#include "nodes/lstm.h"
#include "nodes/matmul.h"
#include "nodes/matmulinteger.h"
#include "nodes/maxpool.h"
#include "nodes/pad.h"
#include "nodes/quantizelinear.h"
#include "nodes/randomuniform.h"
#include "nodes/range.h"
#include "nodes/reduce.h"
#include "nodes/relu.h"
#include "nodes/reshape.h"
#include "nodes/resize.h"
#include "nodes/scatternd.h"
#include "nodes/shape.h"
#include "nodes/slice.h"
#include "nodes/squeeze.h"
#include "nodes/softmax.h"
#include "nodes/split.h"
#include "nodes/transpose.h"
#include "nodes/treeensembleclassifier.h"
#include "nodes/unsqueeze.h"
#include "nodes/upsample.h"
#include "nodes/where.h"

// Create a new onnx2c Node from an operand name of an ONNX Graph node.
// NB: the onnx2c-special graph input and graph output nodes are not created here
Node* Graph::createNode(const onnx::NodeProto &onnx_node)
{
	std::string opName = onnx_node.op_type();
	LOG(TRACE) << "Creating new node: " << onnx_node.name() << std::endl;
	LOG(TRACE) << "     Operand type: " << opName << std::endl;

	if( opName == "Abs" )return new Elementwise("Abs");
	if( opName == "Acos" )return new Elementwise("Acos");
	if( opName == "Acosh" )return new Elementwise("Acosh");
	if( opName == "Add" )return new Elementwise_2("Add");
	if( opName == "And" )return new Elementwise_2("And");
	if( opName == "Asin" )return new Elementwise("Asin");
	if( opName == "Asinh" )return new Elementwise("Asinh");
	if( opName == "Atan" )return new Elementwise("Atan");
	if( opName == "Atanh" )return new Elementwise("Atanh");
	if( opName == "AveragePool" )return new AveragePool;
	if( opName == "BatchNormalization" )return new BatchNormalization;
	if( opName == "BitShift" )return new Elementwise_2("BitShift");
	if( opName == "Cast" )return new Cast;
	if( opName == "Ceil" )return new Elementwise("Ceil");
	if( opName == "Celu" )return new Elementwise("Celu");
	if( opName == "Clip" )return new Clip;
	if( opName == "Concat" )return new Concat;
	if( opName == "Constant" )return new Constant;
	if( opName == "ConstantOfShape" )return new ConstantOfShape;
	if( opName == "Conv" )return new Conv;
	if( opName == "Cos" )return new Elementwise("Cos");
	if( opName == "Cosh" )return new Elementwise("Cosh");
	if( opName == "ConvInteger" )return new ConvInteger;
	if( opName == "ConvTranspose" )return new ConvTranspose;
	if( opName == "DequantizeLinear" )return new DequantizeLinear;
	if( opName == "Div" )return new Elementwise_2("Div");
	if( opName == "Dropout" )return new Dropout;
	if( opName == "DynamicQuantizeLinear" )return new DynamicQuantizeLinear;
	if( opName == "Flatten" )return new Flatten;
	if( opName == "Floor" )return new Elementwise("Floor");
	if( opName == "Elu" )return new Elementwise("Elu");
	if( opName == "Equal")return new Elementwise_2("Equal");
	if( opName == "Erf" )return new Elementwise("Erf");
	if( opName == "Exp" )return new Elementwise("Exp");
	if( opName == "Expand" )return new Expand;
	if( opName == "Gather" )return new Gather;
	if( opName == "Gemm" )return new Gemm;
	if( opName == "GlobalAveragePool" )return new GlobalAveragePool;
	if( opName == "GlobalMaxPool" )return new GlobalMaxPool;
	if( opName == "Greater")return new Elementwise_2("Greater");
	if( opName == "GreaterOrEqual")return new Elementwise_2("GreaterOrEqual");
	if( opName == "HardSigmoid" )return new Elementwise("HardSigmoid");
	if( opName == "HardSwish" )return new Elementwise("HardSwish");
	if( opName == "Identity" )return new Identity;
	if( opName == "InstanceNormalization" )return new InstanceNormalization;
	if( opName == "LayerNormalization" )return new LayerNormalization;
	if( opName == "LeakyRelu" )return new Elementwise("LeakyRelu");
	if( opName == "Less")return new Elementwise_2("Less");
	if( opName == "LessOrEqual")return new Elementwise_2("LessOrEqual");
	if( opName == "Log" )return new Elementwise("Log");
	if( opName == "LogSoftmax" )return new Softmax("LogSoftmax");
	if( opName == "LRN" )return new LRN;
	if( opName == "LSTM" )return new LSTM;
	if( opName == "MatMul" )return new MatMul;
	if( opName == "MatMulInteger" )return new MatMulInteger;
	if( opName == "Max" )return new Elementwise_variadic("Max");
	if( opName == "MaxPool" )return new MaxPool;
	if( opName == "Mean" )return new Elementwise_variadic("Mean");
	if( opName == "Min" )return new Elementwise_variadic("Min");
	if( opName == "Mod" )return new Elementwise_2("Mod");
	if( opName == "Mul" )return new Elementwise_2("Mul");
	if( opName == "Neg" )return new Elementwise("Neg");
	if( opName == "Not" )return new Elementwise("Not");
	if( opName == "Or" )return new Elementwise_2("Or");
	if( opName == "Pad" )return new Pad;
	if( opName == "Pow" )return new Elementwise_2("Pow");
	if( opName == "PRelu" )return new Elementwise_2("PRelu");
	if( opName == "QuantizeLinear" )return new QuantizeLinear;
	if( opName == "RandomUniform" )return new RandomUniform;
	if( opName == "Range" )return new Range;
	if( opName == "ReduceProd" )return new Reduce("Prod");
	if( opName == "ReduceMean" )return new Reduce("Mean");
	if( opName == "ReduceSumSquare" )return new Reduce("SumSquare");
	if( opName == "ReduceMax" )return new Reduce("Max");
	if( opName == "ReduceMin" )return new Reduce("Min");
	if( opName == "ReduceSum" )return new Reduce("Sum");
	if( opName == "ReduceL1" )return new Reduce("L1");
	if( opName == "ReduceL2" )return new Reduce("L2");
	if( opName == "ReduceLogSum" )return new Reduce("LogSum");
	if( opName == "ReduceLogSumExp" )return new Reduce("LogSumExp");
	if( opName == "Reciprocal" )return new Elementwise("Reciprocal");
	if( opName == "Relu" )return new Relu;
	if( opName == "Reshape" )return new Reshape;
	if( opName == "Resize" )return new Resize;
	if( opName == "Round" )return new Elementwise("Round");
	if( opName == "ScatterND" )return new ScatterND;
	if( opName == "Selu" )return new Elementwise("Selu");
	if( opName == "Shape" )return new Shape;
	if( opName == "Shrink" )return new Elementwise("Shrink");
	if( opName == "Sigmoid" )return new Elementwise("Sigmoid");
	if( opName == "Sign" )return new Elementwise("Sign");
	if( opName == "Sin" )return new Elementwise("Sin");
	if( opName == "Sinh" )return new Elementwise("Sinh");
	if( opName == "Slice" )return new Slice;
	if( opName == "Softplus" )return new Elementwise("Softplus");
	if( opName == "Softsign" )return new Elementwise("Softsign");
	if( opName == "Softmax" )return new Softmax("Softmax");
	if( opName == "Split" )return new Split;
	if( opName == "Squeeze" )return new Squeeze;
	if( opName == "Sqrt" )return new Elementwise("Sqrt");
	if( opName == "Sub" )return new Elementwise_2("Sub");
	if( opName == "Sum" )return new Elementwise_variadic("Sum");
	if( opName == "Tan" )return new Elementwise("Tan");
	if( opName == "Tanh" )return new Elementwise("Tanh");
	if( opName == "Transpose" )return new Transpose;
	if( opName == "TreeEnsembleClassifier" )return new TreeEnsembleClassifier();
	if( opName == "ThresholdedRelu" )return new Elementwise("ThresholdedRelu");
	if( opName == "Unsqueeze" )return new Unsqueeze;
	if( opName == "Upsample" )return new Upsample;
	if( opName == "Where" )return new Where;
	if( opName == "Xor" )return new Elementwise_2("Xor");


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
	 *
	 * This updating is needed to mark graph input tensors.
	 * So most of the below is pretty unnecessary?
	 * TODO: clean up
	 * update: graph input tensors are handled better now in processGraph()...
	 *         check where the 'else' branch below triggers.
	 */

	// pointer to the previously existing tensor. This gets updated
	assert(t != nullptr);
	Tensor *prev = findTensorByName(t->name);

	if( prev == NULL ) {
		tensors.push_back(t);
		LOG(DEBUG) << "New tensor: " << t->name << " - "<< t->data_type_str() << " { " << t->str_dimensions() << "}" << std::endl;
		LOG(TRACE) << "    " << t->print_trace_dump();
		// TODO return & remove else {}
	}
	else {
		LOG(TRACE) << "Updating existing tensor: " << t->name << std::endl;
		LOG(TRACE) << "  was: " << prev->print_trace_dump() << std::endl;
		LOG(TRACE) << "  new: " << t->print_trace_dump() << std::endl;

		// if updating an output to be recursive:
		if( t->isRecursive ) {
			// Since this tensor was already added, it was added
			// because it is a graph output.
			// This is because recursion means recursion to same node, not a general loop in the network
			prev->generate = t->generate;
			prev->initialize = t->initialize;
			prev->isRecursive = true;
		}
		// Recursive nodes might need to initialize internal tensors
		if( t->data_buffer )
			prev->data_buffer = t->data_buffer;

		prev->initialize = t->initialize || prev->initialize;
		if( prev->initialize )
			prev->generate=true;

		// huh? what is this use case?
		if( prev->isIO == false )
			prev->isConst = t->isConst;

		// The case where a tensor is marked to be IO.
		// The graph has a lists of input tensors that are input to the first
		// node, not necessarily an input from the user.
		// If the user doesn't provide them, they must be initialized in the graph.
		// E.g. the weights for a Convolution at the start is such an example
		if( t->isIO && prev->initialize == false)
			prev->isIO=true;

		// Some graph IO (output) tensors are not marked with dimensions in ONNX files
		if( prev->rank() == 0 )
			prev->data_dim = t->data_dim;

		LOG(TRACE) << "  now: " << prev->print_trace_dump() << std::endl;
	}
}

Tensor *Graph::findTensor(const std::string &name) const
{
	for( auto o : tensors)
		if( o->name == name )
			return o;
	return NULL;
}

Node* Graph::addGraphInputMetanode()
{
	Node *n = new graph_io();
	n->isResolved = true;
	n->onnx_name = "graph_input";
	nodes.push_back(n);
	return n;
}

Node* Graph::addGraphOutputMetanode()
{
	Node *n = new graph_io();
	n->isResolved = true;
	n->onnx_name = "graph_output";
	nodes.push_back(n);
	return n;
}

Node* Graph::findNodeByName( const std::string node_name )
{
	for( auto n : nodes )
		if( n->onnx_name == node_name )
			return n;
	return nullptr;
}

Tensor* Graph::findTensorByName(std::string name)
{
	for( auto t : tensors)
		if( name == t->name ) {
			return t;
			break;
		}
	return nullptr;
}

