
#include "onnx.pb.h"

#include "node.h"
#include "tensor.h"

/* Command line options */
extern bool quantize;
extern bool target_avr;

namespace toC {

class Graph {
public:
	Graph(
		onnx::ModelProto &onnx_model,
		std::vector<Tensor*> inputs={}
	);

	/* print the entire .h and .cc file contents */
	void print_header(std::ostream &destination);
	void print_source(std::ostream &destination);

	/* print individual parts of the file */
	void print_file_frontmatter(std::ostream &destination);
	void print_global_tensors(std::ostream &destination);
	void print_tensor(const Tensor *, std::ostream &dst);
	void print_functions(std::ostream &destination);
	void print_includes(std::ostream &dst);
	void print_interface_function(std::ostream &dst, bool print_definition=true);

	/* Create the onnx2c graph elements from the ONNX graph */
	void processGraph(
		onnx::ModelProto &onnx_model,
		std::vector<Tensor*> inputs={}
	);
	void resolveGraphNodes(onnx::GraphProto &onnx_graph);

	/* Optimization step: cluster the buffers of intermediate tensors into
	 * unions. This make the memory buffers time shared. */
	void unionize_tensors(void);

	/* Optimization step: Fold Cast-nodes to their predecessor. */
	void fold_casts(void);

	void addInitializedTensor(onnx::TensorProto &tensor);
	Tensor* getIoTensor(onnx::ValueInfoProto &vi);

	void replaceWithQuantized(std::vector<Tensor*> &inputs);
	bool getNodeInputTensors(const onnx::NodeProto &node, toC::Node *inputs);

	bool tryResolveNode(onnx::NodeProto &node);
	bool hasUnresolvedNodes(void);
	Node* createNode(std::string opName);

	int64_t onnx_ir_version(void);
private:
	// The top-level onnx object.
	onnx::ModelProto &model;
	// The tensors of the network. Pointers are added to this
	// vector as walking the graph resolves node outputs.
	// Each node keeps pointers internally also to its own inputs&outputs
	std::vector<Tensor*> tensors;
	// The kernels/nodes/operators of the network.
	std::vector<Node*> nodes;
	Node* findNodeByName( const std::string node_name );

	// Should onnx2c print debug info while compiling
	bool verbose_mode;

	/* Add new tensor to set of known tensors.
	 * If the tensor is not already known (checked by name),
	 * the existing tensor is updated */
	void addTensor(Tensor *t);

	Node* addGraphInputMetanode(void);
	Node* addGraphOutputMetanode(void);


	void log_trace_all_tensors(void)
	{
		LOG(TRACE) << "All known tensors at this moment:" << std::endl;
		for( auto t : tensors ) LOG(TRACE) << "  " << t->print_trace_dump() << std::endl;
	}

	Tensor *findTensor(const std::string &name) const;

	// counter for naming anonymous nodes with a number
	static int anonymous_nodes;

	// For the unionize optimization.
	// TODO: this probably should be in a separate class,
	// design how the data is shared, and possibly write the graph_printer
	// as an optimization class too.
	std::vector<Tensor *> tensor_unions;
	uint32_t add_to_free_union(Tensor *t);
	void mark_union_unoccupied(uint32_t);

};

}

