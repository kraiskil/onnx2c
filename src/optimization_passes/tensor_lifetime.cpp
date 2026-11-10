#include "tensor_lifetime.h"

#include "error.h"
#include "node.h"
#include "tensor.h"

#include <algorithm>
#include <map>
#include <string>

namespace toC {

bool lifetimes_overlap(const TensorLifetime& a, const TensorLifetime& b)
{
	return a.first_use <= b.last_use && b.first_use <= a.last_use;
}

std::vector<TensorLifetime> analyze_tensor_lifetimes(const std::vector<Node*>& nodes, const std::vector<Tensor*>& tensors)
{
	std::map<const Node*, size_t> execution_index;
	size_t next_index = 0;
	for (const Node* node : nodes) {
		if (node->op_name == "graph_io")
			continue;
		execution_index[node] = next_index;
		next_index++;
	}

	std::vector<TensorLifetime> lifetimes;
	for (Tensor* tensor : tensors) {
		if (!tensor->eligible_for_arena())
			continue;
		if (tensor->producer == nullptr)
			ERROR("Tensor " << tensor->name << " is arena-eligible but has no producer");
		auto producer_it = execution_index.find(tensor->producer);
		if (producer_it == execution_index.end())
			ERROR("Tensor " << tensor->name << " producer is not a scheduled compute node");

		TensorLifetime lifetime;
		lifetime.tensor = tensor;
		lifetime.first_use = producer_it->second;
		lifetime.last_use = producer_it->second;
		lifetime.size_bytes = tensor->data_size_bytes();
		lifetime.alignment = tensor->required_alignment();

		for (Node* consumer : tensor->consumers) {
			if (consumer->op_name == "graph_io")
				continue;
			auto consumer_it = execution_index.find(consumer);
			if (consumer_it == execution_index.end())
				ERROR("Tensor " << tensor->name << " has an unscheduled consumer");
			if (consumer_it->second < lifetime.first_use)
				ERROR("Tensor " << tensor->name << " consumer precedes producer");
			lifetime.last_use = std::max(lifetime.last_use, consumer_it->second);
		}

		lifetimes.push_back(lifetime);
	}

	return lifetimes;
}

size_t compute_peak_live_bytes(const std::vector<TensorLifetime>& lifetimes)
{
	size_t max_step = 0;
	for (const TensorLifetime& lifetime : lifetimes)
		max_step = std::max(max_step, lifetime.last_use);

	size_t peak = 0;
	for (size_t step = 0; step <= max_step && !lifetimes.empty(); step++) {
		size_t live_bytes = 0;
		for (const TensorLifetime& lifetime : lifetimes) {
			if (lifetime.first_use <= step && step <= lifetime.last_use) {
				if (lifetime.size_bytes > SIZE_MAX - live_bytes)
					ERROR("Tensor lifetime byte sum overflow while computing peak live bytes");
				live_bytes += lifetime.size_bytes;
			}
		}
		peak = std::max(peak, live_bytes);
	}

	return peak;
}

} // namespace toC
