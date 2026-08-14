/* This file is part of onnx2c.
 *
 * Im2Col optimization pass
 * Transforms Conv operations into fused Im2Col node for better performance.
 * Handles: Conv, depthwise (group>1), and pointwise (1x1) convolutions.
 */

#include "nodes/im2col.h"
#include "graph.h"
#include "nodes/spatialfilter.h"
#include "options.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

using namespace toC;

static bool is_conv_like_node(Node* n)
{
	return n->op_name == "Conv" ||
	       n->op_name == "ConvInteger" ||
	       n->op_name == "QLinearConv";
}

static void replace_consumer(Tensor* tensor, Node* old_consumer, Node* new_consumer)
{
	for (auto& consumer : tensor->consumers) {
		if (consumer == old_consumer) {
			consumer = new_consumer;
			return;
		}
	}
	for (auto consumer : tensor->consumers) {
		if (consumer == new_consumer)
			return;
	}
	tensor->consumers.push_back(new_consumer);
}

static bool is_arithmetic_type(onnx::TensorProto_DataType t)
{
	return t == onnx::TensorProto_DataType_FLOAT ||
	       t == onnx::TensorProto_DataType_DOUBLE ||
	       t == onnx::TensorProto_DataType_FLOAT16 ||
	       t == onnx::TensorProto_DataType_BFLOAT16 ||
	       t == onnx::TensorProto_DataType_INT8 ||
	       t == onnx::TensorProto_DataType_UINT8 ||
	       t == onnx::TensorProto_DataType_INT16 ||
	       t == onnx::TensorProto_DataType_UINT16 ||
	       t == onnx::TensorProto_DataType_INT32 ||
	       t == onnx::TensorProto_DataType_UINT32 ||
	       t == onnx::TensorProto_DataType_INT64 ||
	       t == onnx::TensorProto_DataType_UINT64;
}

static bool is_native_float_type(onnx::TensorProto_DataType t)
{
	return t == onnx::TensorProto_DataType_FLOAT ||
	       t == onnx::TensorProto_DataType_DOUBLE;
}

static bool can_use_explicit_materialized_im2col(Tensor* X, Tensor* W, Tensor* Y)
{
	return is_native_float_type(X->data_type) &&
	       is_native_float_type(W->data_type) &&
	       is_native_float_type(Y->data_type);
}

static bool can_be_im2col(SpatialFilter* conv, Tensor* X, Tensor* W, Tensor* Y)
{
	if (X->rank() != 4 || W->rank() != 4 || Y->rank() != 4)
		return false;

	if (!is_arithmetic_type(X->data_type) ||
	    !is_arithmetic_type(W->data_type) ||
	    !is_arithmetic_type(Y->data_type))
		return false;

	if (conv->group <= 0)
		return false;

	if (X->data_dim[1] % conv->group != 0 || W->data_dim[0] % conv->group != 0)
		return false;

	if (W->data_dim[1] != X->data_dim[1] / conv->group)
		return false;

	return true;
}

struct Im2ColHeuristicParams {
	int64_t tile_m = 16;
	int64_t tile_n = 16;
	int64_t tile_k = 16;
};

enum class ConvImplementation {
	Direct,
	Implicit,
	Explicit,
};

struct Im2ColDecision {
	int64_t m = 0;
	int64_t k = 0;
	int64_t n = 0;
	int64_t input_bytes = 0;
	int64_t output_bytes = 0;
	int64_t weight_bytes = 0;
	int64_t im2col_bytes = 0;
	double expansion_ratio = 0.0;
	double utilization = 0.0;
	int score = 0;
	ConvImplementation implementation = ConvImplementation::Direct;
	std::string reason;
};

static int64_t ceil_to_tile(int64_t value, int64_t tile)
{
	return ((value + tile - 1) / tile) * tile;
}

static const char* implementation_name(ConvImplementation implementation)
{
	switch (implementation) {
		case ConvImplementation::Direct:
			return "direct";
		case ConvImplementation::Implicit:
			return "implicit";
		case ConvImplementation::Explicit:
			return "explicit";
	}
	return "direct";
}

static Im2ColDecision evaluate_conv_implementation_heuristic(
    SpatialFilter* conv, Tensor* X, Tensor* W, Tensor* Y,
    const Im2ColHeuristicParams& params = Im2ColHeuristicParams())
{
	int64_t batch = X->data_dim[0];
	int64_t in_ch = X->data_dim[1];
	int64_t in_h = X->data_dim[2];
	int64_t in_w = X->data_dim[3];
	int64_t out_ch = Y->data_dim[1];
	int64_t out_h = Y->data_dim[2];
	int64_t out_w = Y->data_dim[3];
	int64_t kernel_h = W->data_dim[2];
	int64_t kernel_w = W->data_dim[3];
	int64_t group = conv->group;
	int64_t in_ch_per_group = in_ch / group;

	int64_t kernel_area = kernel_h * kernel_w;
	Im2ColDecision d;
	d.m = out_ch / group;
	d.k = in_ch_per_group * kernel_area;
	d.n = batch * out_h * out_w;
	d.input_bytes = batch * in_ch * in_h * in_w * X->data_elem_size();
	d.output_bytes = batch * out_ch * out_h * out_w * Y->data_elem_size();
	d.weight_bytes = out_ch * in_ch_per_group * kernel_h * kernel_w * W->data_elem_size();
	d.im2col_bytes = d.n * d.k * X->data_elem_size();
	d.expansion_ratio = d.input_bytes > 0 ? (double)d.im2col_bytes / (double)d.input_bytes : INFINITY;

	int64_t tiled_m = ceil_to_tile(d.m, params.tile_m);
	int64_t tiled_n = ceil_to_tile(d.n, params.tile_n);
	int64_t tiled_k = ceil_to_tile(d.k, params.tile_k);
	double tiled_work = (double)tiled_m * (double)tiled_n * (double)tiled_k;
	d.utilization = tiled_work > 0.0 ? ((double)d.m * (double)d.k * (double)d.n) / tiled_work : 0.0;

	bool explicit_supported = can_use_explicit_materialized_im2col(X, W, Y);

	// Performance-oriented implementation selection only. This intentionally
	// does not use im2col workspace size or expansion ratio; memory overhead is
	// reported separately by the benchmark tooling.
	bool depthwise = group == in_ch && group == out_ch;

	if (depthwise) {
		d.implementation = ConvImplementation::Implicit;
		d.reason = "implicit: depthwise Conv is usually fastest without materialized im2col";
		return d;
	}

	if (kernel_area == 1 && d.k <= 32 && d.n >= 3136) {
		d.implementation = ConvImplementation::Direct;
		d.reason = "direct: low-K 1x1 large-spatial Conv is faster without im2col lowering";
		return d;
	}

	if (explicit_supported) {
		d.implementation = ConvImplementation::Explicit;
		d.reason = "explicit: non-depthwise Conv favors materialized im2col GEMM";
		return d;
	}

	d.implementation = ConvImplementation::Implicit;
	d.reason = "implicit: explicit materialization is unsupported for this Conv arithmetic";
	return d;
}

static void log_im2col_decision(SpatialFilter* conv, Tensor* X, Tensor* W, Tensor* Y, const Im2ColDecision& d)
{
	LOG(DEBUG) << "  im2col decision:"
	           << " layer=" << conv->onnx_name
	           << " input=[" << X->data_dim[0] << "," << X->data_dim[1] << "," << X->data_dim[2] << "," << X->data_dim[3] << "]"
	           << " output=[" << Y->data_dim[0] << "," << Y->data_dim[1] << "," << Y->data_dim[2] << "," << Y->data_dim[3] << "]"
	           << " kernel=[" << W->data_dim[2] << "," << W->data_dim[3] << "]"
	           << " groups=" << conv->group
	           << " M=" << d.m
	           << " N=" << d.n
	           << " K=" << d.k
	           << " im2col_bytes=" << d.im2col_bytes
	           << " expansion_ratio=" << d.expansion_ratio
	           << " utilization=" << d.utilization
	           << " score=" << d.score
	           << " implementation=" << implementation_name(d.implementation)
	           << " selected=" << (d.implementation == ConvImplementation::Direct ? "no" : "yes")
	           << " reason=\"" << d.reason << "\""
	           << std::endl;
}

static Im2Col::ArithmeticMode arithmetic_mode_for(Node* n)
{
	if (n->op_name == "ConvInteger")
		return Im2Col::ConvInteger;
	if (n->op_name == "QLinearConv")
		return Im2Col::QLinearConv;
	return Im2Col::Conv;
}

void Graph::im2col(void)
{
	LOG(DEBUG) << "Optimisation pass: im2col" << std::endl;

	std::vector<Node*> candidates;
	std::unordered_map<Node*, ConvImplementation> implementations;

	LOG(TRACE) << "im2col pass: entered" << std::endl;

	int nodes_before = nodes.size();
	int tensors_before = tensors.size();
	LOG(TRACE) << "im2col pass: nodes=" << nodes_before << " tensors=" << tensors_before << std::endl;

	for (auto n : nodes) {
		if (!is_conv_like_node(n))
			continue;

		LOG(DEBUG) << "Considering im2col for node: " << n->onnx_name << " (op: " << n->op_name << ")" << std::endl;

		SpatialFilter* conv = dynamic_cast<SpatialFilter*>(n);
		if (!conv) {
			LOG(WARNING) << "  Skipping: not a SpatialFilter" << std::endl;
			continue;
		}

		Tensor* X = const_cast<Tensor*>(conv->get_X());
		Tensor* W = const_cast<Tensor*>(conv->get_W());
		Tensor* Y = conv->get_output_tensor(0);
		LOG(TRACE) << "  Input tensor: " << X->name << " rank=" << X->rank() << std::endl;
		if (!can_be_im2col(conv, X, W, Y)) {
			LOG(WARNING) << "  Skipping: unsupported Conv for im2col" << std::endl;
			continue;
		}

		ConvImplementation implementation = ConvImplementation::Direct;
		if (options.opt_im2col_mode == im2col_mode::HEURISTIC) {
			Im2ColDecision decision = evaluate_conv_implementation_heuristic(conv, X, W, Y);
			log_im2col_decision(conv, X, W, Y, decision);
			implementation = decision.implementation;
			if (implementation == ConvImplementation::Direct) {
				LOG(DEBUG) << "  Selected implementation: direct Conv" << std::endl;
				continue;
			}
			LOG(DEBUG) << "  Selected implementation: " << implementation_name(implementation) << " im2col" << std::endl;
		}
		else if (options.opt_im2col_mode == im2col_mode::IMPLICIT) {
			implementation = ConvImplementation::Implicit;
			LOG(DEBUG) << "  Selected implementation: implicit/fused im2col (forced)" << std::endl;
		}
		else if (options.opt_im2col_mode == im2col_mode::EXPLICIT) {
			implementation = can_use_explicit_materialized_im2col(X, W, Y) ? ConvImplementation::Explicit : ConvImplementation::Implicit;
			LOG(DEBUG) << "  Selected implementation: explicit/materialized im2col (forced where supported)" << std::endl;
		}
		else {
			continue;
		}

		implementations[n] = implementation;
		candidates.push_back(n);
	}

	for (auto n : candidates) {
		SpatialFilter* conv = dynamic_cast<SpatialFilter*>(n);

		Tensor* X = const_cast<Tensor*>(conv->get_X());
		Tensor* W = const_cast<Tensor*>(conv->get_W());
		Tensor* Y = conv->get_output_tensor(0);

		Im2Col* im2col = new Im2Col;
		im2col->onnx_name = conv->onnx_name + "_im2col";
		im2col->isResolved = true;
		im2col->arithmetic_mode = arithmetic_mode_for(n);
		im2col->implementation = implementations[n] == ConvImplementation::Explicit ? Im2Col::Explicit : Im2Col::Implicit;

		im2col->batch = X->data_dim[0];
		im2col->in_ch = X->data_dim[1];
		im2col->in_h = X->data_dim[2];
		im2col->in_w = X->data_dim[3];

		im2col->out_ch = W->data_dim[0];
		im2col->kernel_h = W->data_dim[2];
		im2col->kernel_w = W->data_dim[3];

		im2col->out_h = Y->data_dim[2];
		im2col->out_w = Y->data_dim[3];

		im2col->group = conv->group;

		if (conv->strides.size() >= 2) {
			im2col->stride_h = conv->strides[0];
			im2col->stride_w = conv->strides[1];
		}
		if (conv->pads.size() >= 4) {
			im2col->pad_h = conv->pads[0];
			im2col->pad_w = conv->pads[1];
		}
		if (conv->dilations.size() >= 2) {
			im2col->dilation_h = conv->dilations[0];
			im2col->dilation_w = conv->dilations[1];
		}

		if (im2col->arithmetic_mode == Im2Col::Conv) {
			im2col->has_bias = (conv->get_number_of_inputs() >= 3);
			im2col->register_input(X, "x");
			im2col->register_input(W, "w");
			if (im2col->has_bias) {
				Tensor* B = conv->get_input_tensor(2);
				im2col->register_input(B, "bias");
				replace_consumer(B, n, im2col);
			}
		}
		else if (im2col->arithmetic_mode == Im2Col::ConvInteger) {
			im2col->has_x_zero_point = (conv->get_number_of_inputs() >= 3);
			im2col->has_w_zero_point = (conv->get_number_of_inputs() >= 4);
			im2col->register_input(X, "x");
			im2col->register_input(W, "w");
			if (im2col->has_x_zero_point) {
				Tensor* X_zero_point = conv->get_input_tensor(2);
				im2col->register_input(X_zero_point, "x_zero_point");
				replace_consumer(X_zero_point, n, im2col);
			}
			if (im2col->has_w_zero_point) {
				Tensor* W_zero_point = conv->get_input_tensor(3);
				im2col->register_input(W_zero_point, "w_zero_point");
				replace_consumer(W_zero_point, n, im2col);
			}
		}
		else {
			im2col->has_bias = (conv->get_number_of_inputs() >= 9);
			im2col->register_input(X, "x");
			im2col->register_input(conv->get_input_tensor(1), "x_scale");
			im2col->register_input(conv->get_input_tensor(2), "x_zero_point");
			im2col->register_input(W, "w");
			im2col->register_input(conv->get_input_tensor(4), "w_scale");
			im2col->register_input(conv->get_input_tensor(5), "w_zero_point");
			im2col->register_input(conv->get_input_tensor(6), "y_scale");
			im2col->register_input(conv->get_input_tensor(7), "y_zero_point");
			for (unsigned i = 1; i <= 7; i++)
				replace_consumer(conv->get_input_tensor(i), n, im2col);
			if (im2col->has_bias) {
				Tensor* B = conv->get_input_tensor(8);
				im2col->register_input(B, "bias");
				replace_consumer(B, n, im2col);
			}
		}
		im2col->register_output(Y, "y");

		replace_consumer(X, n, im2col);
		replace_consumer(W, n, im2col);

		auto it = std::find(nodes.begin(), nodes.end(), n);
		if (it != nodes.end()) {
			*it = im2col;
		}
		std::string transformed_node_name = n->onnx_name;
		delete n;

		LOG(TRACE) << "Transform complete for " << transformed_node_name << std::endl;
	}

	int nodes_after = nodes.size();
	int tensors_after = tensors.size();
	LOG(TRACE) << "im2col pass: after: nodes=" << nodes_after << " tensors=" << tensors_after << std::endl;

	LOG(TRACE) << "im2col pass finished" << std::endl;
}
