#pragma once

#include "tensor_lifetime.h"
#include <cstddef>
#include <string>
#include <vector>

namespace toC {

class Tensor;

struct ArenaAllocation {
	Tensor* tensor = nullptr;
	size_t offset = 0;
	size_t size = 0;
	size_t alignment = 1;
	size_t first_use = 0;
	size_t last_use = 0;
};

struct ArenaPlan {
	std::vector<ArenaAllocation> allocations;
	size_t arena_size = 0;
};

struct TensorArenaMetrics {
	size_t eligible_tensor_count = 0;
	size_t total_intermediate_bytes = 0;
	size_t peak_live_lower_bound = 0;
	size_t union_baseline_bytes = 0;
	size_t arena_bytes = 0;
	std::string placement_strategy;
	std::string schedule_strategy;
};

enum class arena_strategy { first_fit,
	                    memory_schedule };
arena_strategy parse_arena_strategy(const std::string& value);
const char* arena_strategy_name(arena_strategy strategy);

size_t align_up(size_t value, size_t alignment);
bool address_ranges_overlap(size_t offset_a, size_t size_a, size_t offset_b, size_t size_b);
bool allocations_conflict(const ArenaAllocation& a, const ArenaAllocation& b);
bool validate_arena_plan(const ArenaPlan& plan, const std::vector<TensorLifetime>& lifetimes, std::string* reason = nullptr);

class TensorArenaPlanner {
	public:
	ArenaPlan plan(const std::vector<TensorLifetime>& lifetimes) const;
	ArenaPlan plan(const std::vector<TensorLifetime>& lifetimes, arena_strategy strategy, TensorArenaMetrics* metrics = nullptr) const;
};

} // namespace toC
