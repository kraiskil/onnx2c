#pragma once

#include <cstddef>
#include <vector>

namespace toC {

class Node;
class Tensor;

struct TensorLifetime {
	Tensor* tensor = nullptr;
	size_t first_use = 0;
	size_t last_use = 0;
	size_t size_bytes = 0;
	size_t alignment = 1;
};

bool lifetimes_overlap(const TensorLifetime& a, const TensorLifetime& b);
std::vector<TensorLifetime> analyze_tensor_lifetimes(const std::vector<Node*>& nodes, const std::vector<Tensor*>& tensors);
size_t compute_peak_live_bytes(const std::vector<TensorLifetime>& lifetimes);

} // namespace toC
