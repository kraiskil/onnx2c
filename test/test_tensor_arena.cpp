#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include "catch.hpp"

#include "graph.h"
#include "options.h"
#include "optimization_passes/tensor_arena_planner.h"
#include "tensor.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <vector>

using namespace toC;

struct onnx2c_opts options;

namespace {

std::unique_ptr<Tensor> make_tensor(const std::string& name, onnx::TensorProto_DataType type, const std::vector<int>& dims)
{
	auto tensor = std::make_unique<Tensor>();
	tensor->name = name;
	tensor->data_type = type;
	tensor->data_dim = dims;
	return tensor;
}

TensorLifetime make_lifetime(Tensor* tensor, size_t first_use, size_t last_use)
{
	return TensorLifetime{tensor, first_use, last_use, tensor->data_size_bytes(), tensor->required_alignment()};
}

const ArenaAllocation& find_allocation(const ArenaPlan& plan, const std::string& tensor_name)
{
	auto it = std::find_if(plan.allocations.begin(), plan.allocations.end(), [&](const ArenaAllocation& allocation) {
		return allocation.tensor->name == tensor_name;
	});
	REQUIRE(it != plan.allocations.end());
	return *it;
}

std::vector<std::pair<std::string, size_t>> summarize_plan(const ArenaPlan& plan)
{
	std::vector<std::pair<std::string, size_t>> summary;
	for (const ArenaAllocation& allocation : plan.allocations)
		summary.emplace_back(allocation.tensor->name, allocation.offset);
	std::sort(summary.begin(), summary.end());
	return summary;
}

void reset_options()
{
	options = onnx2c_opts();
}

onnx::ValueInfoProto make_value_info(const std::string& name, onnx::TensorProto_DataType type, const std::vector<int64_t>& dims)
{
	onnx::ValueInfoProto value;
	value.set_name(name);
	auto* tensor_type = value.mutable_type()->mutable_tensor_type();
	tensor_type->set_elem_type(type);
	auto* shape = tensor_type->mutable_shape();
	for (int64_t dim : dims)
		shape->add_dim()->set_dim_value(dim);
	return value;
}

onnx::TensorProto make_int64_initializer(const std::string& name, const std::vector<int64_t>& dims, const std::vector<int64_t>& values)
{
	onnx::TensorProto tensor;
	tensor.set_name(name);
	tensor.set_data_type(onnx::TensorProto_DataType_INT64);
	for (int64_t dim : dims)
		tensor.add_dims(dim);
	for (int64_t value : values)
		tensor.add_int64_data(value);
	return tensor;
}

onnx::ModelProto make_model_linear()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {1, 4});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {1, 4});

	auto* relu0 = graph->add_node();
	relu0->set_op_type("Relu");
	relu0->set_name("relu0");
	relu0->add_input("input");
	relu0->add_output("a");

	auto* relu1 = graph->add_node();
	relu1->set_op_type("Relu");
	relu1->set_name("relu1");
	relu1->add_input("a");
	relu1->add_output("b");

	auto* relu2 = graph->add_node();
	relu2->set_op_type("Relu");
	relu2->set_name("relu2");
	relu2->add_input("b");
	relu2->add_output("output");
	return model;
}

onnx::ModelProto make_model_fork_join()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {1, 4});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {1, 4});

	auto* relu_left = graph->add_node();
	relu_left->set_op_type("Relu");
	relu_left->set_name("relu_left");
	relu_left->add_input("input");
	relu_left->add_output("left");

	auto* relu_right = graph->add_node();
	relu_right->set_op_type("Relu");
	relu_right->set_name("relu_right");
	relu_right->add_input("input");
	relu_right->add_output("right");

	auto* add = graph->add_node();
	add->set_op_type("Add");
	add->set_name("add_out");
	add->add_input("left");
	add->add_input("right");
	add->add_output("output");
	return model;
}

onnx::ModelProto make_model_residual()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {1, 4});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {1, 4});

	auto* skip = graph->add_node();
	skip->set_op_type("Identity");
	skip->set_name("skip");
	skip->add_input("input");
	skip->add_output("skip_tensor");

	auto* relu0 = graph->add_node();
	relu0->set_op_type("Relu");
	relu0->set_name("relu0");
	relu0->add_input("input");
	relu0->add_output("mid0");

	auto* relu1 = graph->add_node();
	relu1->set_op_type("Relu");
	relu1->set_name("relu1");
	relu1->add_input("mid0");
	relu1->add_output("mid1");

	auto* add = graph->add_node();
	add->set_op_type("Add");
	add->set_name("add_out");
	add->add_input("mid1");
	add->add_input("skip_tensor");
	add->add_output("output");
	return model;
}

onnx::ModelProto make_model_scheduler_branch()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {1, 100});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {});
	auto add_relu = [&](const char* name, const char* input, const char* output) {
		auto* node = graph->add_node(); node->set_op_type("Relu"); node->set_name(name); node->add_input(input); node->add_output(output);
	};
	add_relu("branch_b", "input", "b");
	add_relu("branch_c", "input", "c");
	auto* reduce_b = graph->add_node(); reduce_b->set_op_type("ReduceMean"); reduce_b->set_name("reduce_b"); reduce_b->add_input("b"); reduce_b->add_output("d");
	// The resolver accepts the default axes and keepdims for this scalar reduction.
	auto* reduce_c = graph->add_node(); reduce_c->set_op_type("ReduceMean"); reduce_c->set_name("reduce_c"); reduce_c->add_input("c"); reduce_c->add_output("e");
	auto* add = graph->add_node(); add->set_op_type("Add"); add->set_name("join"); add->add_input("d"); add->add_input("e"); add->add_output("output");
	return model;
}

onnx::ModelProto make_model_scalar()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {});

	auto* abs = graph->add_node();
	abs->set_op_type("Abs");
	abs->set_name("abs0");
	abs->add_input("input");
	abs->add_output("mid");

	auto* identity = graph->add_node();
	identity->set_op_type("Identity");
	identity->set_name("identity0");
	identity->add_input("mid");
	identity->add_output("output");
	return model;
}

onnx::ModelProto make_model_no_intermediate()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input", onnx::TensorProto_DataType_FLOAT, {1, 4});
	*graph->add_output() = make_value_info("output", onnx::TensorProto_DataType_FLOAT, {1, 4});

	auto* identity = graph->add_node();
	identity->set_op_type("Identity");
	identity->set_name("identity0");
	identity->add_input("input");
	identity->add_output("output");
	return model;
}

onnx::ModelProto make_model_arena_advantage()
{
	onnx::ModelProto model;
	model.mutable_opset_import()->Add()->set_version(13);
	auto* graph = model.mutable_graph();
	*graph->add_input() = make_value_info("input_large", onnx::TensorProto_DataType_FLOAT, {250});
	*graph->add_input() = make_value_info("input_small", onnx::TensorProto_DataType_FLOAT, {225});
	*graph->add_output() = make_value_info("large_out", onnx::TensorProto_DataType_FLOAT, {250});
	*graph->add_output() = make_value_info("small_out", onnx::TensorProto_DataType_FLOAT, {225});
	*graph->add_initializer() = make_int64_initializer("split_sizes", {2}, {100, 125});

	auto* identity = graph->add_node();
	identity->set_op_type("Identity");
	identity->set_name("identity_large");
	identity->add_input("input_large");
	identity->add_output("large_tmp");

	auto* output_id = graph->add_node();
	output_id->set_op_type("Identity");
	output_id->set_name("identity_out");
	output_id->add_input("large_tmp");
	output_id->add_output("large_out");

	auto* split = graph->add_node();
	split->set_op_type("Split");
	split->set_name("split_small");
	split->add_input("input_small");
	split->add_input("split_sizes");
	split->add_output("branch0");
	split->add_output("branch1");
	auto* axis = split->add_attribute();
	axis->set_name("axis");
	axis->set_type(onnx::AttributeProto_AttributeType_INT);
	axis->set_i(0);

	auto* concat = graph->add_node();
	concat->set_op_type("Concat");
	concat->set_name("concat_small");
	concat->add_input("branch0");
	concat->add_input("branch1");
	concat->add_output("small_out");
	auto* concat_axis = concat->add_attribute();
	concat_axis->set_name("axis");
	concat_axis->set_type(onnx::AttributeProto_AttributeType_INT);
	concat_axis->set_i(0);
	return model;
}

std::string render_source(onnx::ModelProto& model, bool no_globals = false)
{
	reset_options();
	Graph graph(model);
	graph.assign_tensor_memory_arena();
	graph.set_no_globals(no_globals);
	std::ostringstream source;
	graph.print_source(source, "entry");
	return source.str();
}

TEST_CASE("memory scheduler preserves dependencies and is deterministic", "[tensor_arena][scheduler]")
{
	onnx::ModelProto model = make_model_fork_join();
	reset_options();
	Graph graph(model);
	auto original = graph.execution_node_names_for_test();
	{
		Graph repeat(model);
		repeat.apply_memory_schedule_for_test();
		auto first = repeat.execution_node_names_for_test();
		Graph again(model);
		again.apply_memory_schedule_for_test();
		REQUIRE(first == again.execution_node_names_for_test());
		REQUIRE(first.size() == original.size());
	}
}

TEST_CASE("memory scheduler leaves a linear graph unchanged", "[tensor_arena][scheduler]")
{
	onnx::ModelProto model = make_model_linear();
	reset_options();
	Graph graph(model);
	auto original = graph.execution_node_names_for_test();
	{
		Graph candidate(model);
		candidate.apply_memory_schedule_for_test();
		REQUIRE(candidate.execution_node_names_for_test() == original);
	}
}

TEST_CASE("scheduler keeps original order for equal ready choices", "[tensor_arena][scheduler]")
{
	onnx::ModelProto model = make_model_fork_join();
	reset_options();
	Graph graph(model);
	graph.apply_memory_schedule_for_test();
	REQUIRE(graph.execution_node_names_for_test() == std::vector<std::string>{"relu_left", "relu_right", "add_out"});
}

TEST_CASE("minimum projected live scheduler reduces an exact branch peak", "[tensor_arena][scheduler]")
{
	onnx::ModelProto model = make_model_scheduler_branch();
	reset_options();
	Graph original(model);
	auto original_names = original.execution_node_names_for_test();
	auto original_lifetimes = original.analyze_tensor_lifetimes();
	REQUIRE(original_names == std::vector<std::string>{"branch_b", "branch_c", "reduce_b", "reduce_c", "join"});
	REQUIRE(compute_peak_live_bytes(original_lifetimes) == 804);

	Graph scheduled(model);
	scheduled.apply_memory_schedule_for_test();
	REQUIRE(scheduled.execution_node_names_for_test() == std::vector<std::string>{"branch_b", "reduce_b", "branch_c", "reduce_c", "join"});
	auto lifetimes = scheduled.analyze_tensor_lifetimes();
	REQUIRE(compute_peak_live_bytes(lifetimes) == 408);
	for (const auto& lifetime : lifetimes)
		REQUIRE(lifetime.first_use <= lifetime.last_use);
}

} // namespace

TEST_CASE("planner sequential full reuse", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {100});
	auto b = make_tensor("B", onnx::TensorProto_DataType_UINT8, {100});
	TensorArenaPlanner planner;
	ArenaPlan plan = planner.plan({make_lifetime(a.get(), 0, 1), make_lifetime(b.get(), 2, 3)});
	REQUIRE(plan.arena_size == 100);
	REQUIRE(find_allocation(plan, "A").offset == 0);
	REQUIRE(find_allocation(plan, "B").offset == 0);
}

TEST_CASE("planner concurrent tensors do not overlap", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {100});
	auto b = make_tensor("B", onnx::TensorProto_DataType_UINT8, {100});
	TensorArenaPlanner planner;
	ArenaPlan plan = planner.plan({make_lifetime(a.get(), 0, 3), make_lifetime(b.get(), 1, 2)});
	REQUIRE(plan.arena_size >= 200);
	REQUIRE(address_ranges_overlap(find_allocation(plan, "A").offset, 100, find_allocation(plan, "B").offset, 100) == false);
}

TEST_CASE("planner same-node boundary lifetimes overlap", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {100});
	auto b = make_tensor("B", onnx::TensorProto_DataType_UINT8, {100});
	TensorArenaPlanner planner;
	ArenaPlan plan = planner.plan({make_lifetime(a.get(), 0, 2), make_lifetime(b.get(), 2, 4)});
	REQUIRE(plan.arena_size >= 200);
	REQUIRE(find_allocation(plan, "A").offset != find_allocation(plan, "B").offset);
}

TEST_CASE("planner canonical partial reuse beats unions", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {1000});
	auto b = make_tensor("B", onnx::TensorProto_DataType_UINT8, {400});
	auto c = make_tensor("C", onnx::TensorProto_DataType_UINT8, {500});
	TensorArenaPlanner planner;
	ArenaPlan plan = planner.plan({make_lifetime(a.get(), 0, 2), make_lifetime(b.get(), 3, 6), make_lifetime(c.get(), 3, 6)});
	REQUIRE(plan.arena_size == 1000);
	REQUIRE(find_allocation(plan, "A").offset == 0);
	REQUIRE(find_allocation(plan, "C").offset == 0);
	REQUIRE(find_allocation(plan, "B").offset == 500);
}

TEST_CASE("planner respects alignment and deterministic order", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {3});
	auto b = make_tensor("B", onnx::TensorProto_DataType_FLOAT, {2});
	auto c = make_tensor("C", onnx::TensorProto_DataType_UINT16, {3});
	std::vector<TensorLifetime> lifetimes = {
	    make_lifetime(a.get(), 0, 1),
	    make_lifetime(b.get(), 2, 4),
	    make_lifetime(c.get(), 2, 4),
	};
	TensorArenaPlanner planner;
	ArenaPlan first = planner.plan(lifetimes);
	std::reverse(lifetimes.begin(), lifetimes.end());
	ArenaPlan second = planner.plan(lifetimes);
	REQUIRE(find_allocation(first, "B").offset % alignof(float) == 0);
	REQUIRE(find_allocation(first, "C").offset % alignof(uint16_t) == 0);
	REQUIRE(summarize_plan(first) == summarize_plan(second));
}

TEST_CASE("planner supports scalars and mixed data types", "[tensor_arena][planner]")
{
	auto scalar = make_tensor("scalar", onnx::TensorProto_DataType_DOUBLE, {});
	auto bytes = make_tensor("bytes", onnx::TensorProto_DataType_BOOL, {7});
	TensorArenaPlanner planner;
	ArenaPlan plan = planner.plan({make_lifetime(scalar.get(), 0, 1), make_lifetime(bytes.get(), 2, 3)});
	REQUIRE(find_allocation(plan, "scalar").size == sizeof(double));
	REQUIRE(find_allocation(plan, "scalar").offset % alignof(double) == 0);
	REQUIRE(plan.arena_size >= sizeof(double));
}

TEST_CASE("arena validator rejects invalid plans", "[tensor_arena][planner]")
{
	auto a = make_tensor("A", onnx::TensorProto_DataType_UINT8, {16});
	auto b = make_tensor("B", onnx::TensorProto_DataType_UINT8, {16});
	std::vector<TensorLifetime> lifetimes = {make_lifetime(a.get(), 0, 2), make_lifetime(b.get(), 1, 3)};
	ArenaPlan overlap_plan;
	overlap_plan.arena_size = 16;
	overlap_plan.allocations = {
	    ArenaAllocation{a.get(), 0, 16, 1, 0, 2},
	    ArenaAllocation{b.get(), 0, 16, 1, 1, 3},
	};
	std::string reason;
	REQUIRE(validate_arena_plan(overlap_plan, lifetimes, &reason) == false);

	ArenaPlan misaligned_plan;
	misaligned_plan.arena_size = 16;
	misaligned_plan.allocations = {ArenaAllocation{a.get(), 1, 16, 2, 0, 2}};
	REQUIRE(validate_arena_plan(misaligned_plan, {make_lifetime(a.get(), 0, 2)}, &reason) == false);

	ArenaPlan duplicate_plan;
	duplicate_plan.arena_size = 32;
	duplicate_plan.allocations = {
	    ArenaAllocation{a.get(), 0, 16, 1, 0, 2},
	    ArenaAllocation{a.get(), 16, 16, 1, 0, 2},
	};
	REQUIRE(validate_arena_plan(duplicate_plan, {make_lifetime(a.get(), 0, 2)}, &reason) == false);

	ArenaPlan overflow_plan;
	overflow_plan.arena_size = 8;
	overflow_plan.allocations = {ArenaAllocation{a.get(), SIZE_MAX - 3, 16, 1, 0, 2}};
	REQUIRE(validate_arena_plan(overflow_plan, {make_lifetime(a.get(), 0, 2)}, &reason) == false);
}

TEST_CASE("graph arena codegen emits global arena and aliases", "[tensor_arena][codegen]")
{
	onnx::ModelProto model = make_model_linear();
	std::string source = render_source(model, false);
	REQUIRE(source.find("tensor_arena_storage") != std::string::npos);
	REQUIRE(source.find("#define tensor_a") != std::string::npos);
	REQUIRE(source.find("#define tensor_b") != std::string::npos);
	REQUIRE(source.find("union tensor_union_") == std::string::npos);
	REQUIRE(source.find("static float tensor_a") == std::string::npos);
	REQUIRE(source.find("static float tensor_b") == std::string::npos);
}

TEST_CASE("graph arena codegen handles fork join and residual lifetimes", "[tensor_arena][codegen]")
{
	{
		onnx::ModelProto model = make_model_fork_join();
		reset_options();
		Graph graph(model);
		graph.assign_tensor_memory_arena();
		auto metrics = graph.get_tensor_memory_metrics();
		REQUIRE(metrics.arena_bytes >= 32);
	}
	{
		onnx::ModelProto model = make_model_residual();
		reset_options();
		Graph graph(model);
		graph.assign_tensor_memory_arena();
		auto metrics = graph.get_tensor_memory_metrics();
		REQUIRE(metrics.arena_bytes >= 32);
	}
}

TEST_CASE("graph arena reports strict improvement over union baseline", "[tensor_arena][codegen]")
{
	onnx::ModelProto model = make_model_arena_advantage();
	reset_options();
	Graph graph(model);
	graph.assign_tensor_memory_arena();
	const TensorArenaMetrics& metrics = graph.get_tensor_memory_metrics();
	REQUIRE(metrics.union_baseline_bytes == 1500);
	REQUIRE(metrics.arena_bytes == 1000);
	REQUIRE(metrics.arena_bytes < metrics.union_baseline_bytes);
	std::string source = render_source(model, false);
	REQUIRE(source.find("data_float[") != std::string::npos);
}

TEST_CASE("graph arena codegen supports scalars multidimensional tensors and no-globals", "[tensor_arena][codegen]")
{
	{
		onnx::ModelProto model = make_model_scalar();
		std::string source = render_source(model, false);
		REQUIRE(source.find("tensor_arena_storage.data_float[") != std::string::npos);
	}
	{
		onnx::ModelProto model = make_model_linear();
		std::string source = render_source(model, true);
		REQUIRE(source.find("static union {\n    uint8_t data[") == std::string::npos);
		REQUIRE(source.find("\tunion {\n\t\tfloat data_float[") != std::string::npos);
	}
}

TEST_CASE("graph arena codegen omits arena when no eligible intermediates exist", "[tensor_arena][codegen]")
{
	onnx::ModelProto model = make_model_no_intermediate();
	reset_options();
	Graph graph(model);
	graph.assign_tensor_memory_arena();
	std::ostringstream source;
	graph.print_source(source, "entry");
	REQUIRE(graph.uses_tensor_arena() == false);
	REQUIRE(source.str().find("tensor_arena_storage") == std::string::npos);
}
