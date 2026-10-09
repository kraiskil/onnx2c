#pragma once
#include "onnx.pb.h"

#include "node.h"
#include "optimization_passes/tensor_arena_planner.h"
#include "optimization_passes/tensor_lifetime.h"
#include "tensor.h"

/* Command line options */
extern bool target_avr;

namespace toC {

class Graph {
	public:
	Graph(
	    onnx::ModelProto& onnx_model,
	    std::vector<Tensor*> inputs = {});

	/* print the entire .h and .cc file contents */
	void print_header(std::ostream& destination, const std::string& interface_func_name);
	void print_source(std::ostream& destination, const std::string& interface_func_name);
	void print_initialization(std::ostream& destination);
	void apply_memory_schedule_for_test();
	std::vector<std::string> execution_node_names_for_test() const;

	/* print individual parts of the file */
	void print_file_frontmatter(std::ostream& destination);
	void print_global_tensors(std::ostream& destination);
	void print_arena_storage_members(std::ostream& destination);
	void print_tensor(const Tensor*, std::ostream& dst);
	void print_functions(std::ostream& destination);
	void print_includes(std::ostream& dst);
	void print_interface_function(std::ostream& dst, bool print_definition = true, const std::string& func_name = "entry");

	/* Create the onnx2c graph elements from the ONNX graph */
	void processGraph(
	    onnx::ModelProto& onnx_model,
	    std::vector<Tensor*> inputs = {});
	void resolveGraphNodes(onnx::GraphProto& onnx_graph);

	/* Optimization step: cluster the buffers of intermediate tensors into
	 * unions. This make the memory buffers time shared. */
	void unionize_tensors(void);

	void clear_tensor_memory_assignments(void);
	void assign_tensor_memory_none(void);
	void assign_tensor_memory_union(void);
	void assign_tensor_memory_arena(void);
	std::vector<TensorLifetime> analyze_tensor_lifetimes(void) const;
	const ArenaPlan& get_tensor_arena_plan(void) const { return tensor_arena_plan; }
	const TensorArenaMetrics& get_tensor_memory_metrics(void) const { return tensor_memory_metrics; }
	bool uses_tensor_arena(void) const { return tensor_arena_enabled && tensor_arena_plan.arena_size > 0; }

	/* Optimization step: Fold Cast-nodes to their predecessor. */
	void fold_casts(void);

	/* Set print options */
	void set_no_globals(bool ng) { no_globals = ng; }

	void addInitializedTensor(onnx::TensorProto& tensor);
	Tensor* getIoTensor(onnx::ValueInfoProto& vi);

	void replaceWithQuantized(std::vector<Tensor*>& inputs);
	bool getNodeInputTensors(const onnx::NodeProto& node, toC::Node* inputs);

	bool tryResolveNode(onnx::NodeProto& node);
	bool hasUnresolvedNodes(void);
	Node* createNode(const onnx::NodeProto& node);

	int64_t onnx_ir_version(void);

	private:
	// The top-level onnx object.
	onnx::ModelProto& model;

	// The tensors of the network. Pointers are added to this
	// vector as walking the graph resolves node outputs.
	// Each node keeps pointers internally also to its own inputs&outputs
	std::vector<Tensor*> tensors;
	// Return the requested tensor.
	// nullptr if the tensor is unknown.
	Tensor* findTensorByName(std::string name);

	// The kernels/nodes/operators of the network.
	std::vector<Node*> nodes;
	Node* findNodeByName(const std::string node_name);

	// Should onnx2c print debug info while compiling
	bool verbose_mode;

	/* Add new tensor to set of known tensors.
	 * If the tensor is not already known (checked by name),
	 * the existing tensor is updated */
	void addTensor(Tensor* t);

	Node* addGraphInputMetanode(void);
	Node* addGraphOutputMetanode(void);

	void log_trace_all_tensors(void)
	{
		LOG(TRACE) << "All known tensors at this moment:" << std::endl;
		for (auto t : tensors)
			LOG(TRACE) << "  " << t->print_trace_dump() << std::endl;
	}

	Tensor* findTensor(const std::string& name) const;
	ArenaPlan build_union_baseline_arena_plan(const std::vector<TensorLifetime>& lifetimes) const;
	void log_tensor_arena_metrics(void) const;
	void schedule_nodes_for_memory();

	// counter for naming anonymous nodes with a number
	static int anonymous_nodes;

	// For the unionize optimization.
	// TODO: this probably should be in a separate class,
	// design how the data is shared, and possibly write the graph_printer
	// as an optimization class too.
	std::vector<Tensor*> tensor_unions;
	uint32_t add_to_free_union(Tensor* t);
	void mark_union_unoccupied(uint32_t);

	ArenaPlan tensor_arena_plan;
	TensorArenaMetrics tensor_memory_metrics;
	bool tensor_arena_enabled = false;

	// Print options
	bool no_globals = false;
};

} // namespace toC
