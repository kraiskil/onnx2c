#include "tensor_arena_planner.h"

#include "error.h"
#include "tensor.h"

#include <algorithm>
#include <map>
#include <set>
#include <limits>

namespace toC {
namespace {

size_t checked_add(size_t a, size_t b, const char* context)
{
	if (a > SIZE_MAX - b)
		ERROR(context);
	return a + b;
}

} // namespace

size_t align_up(size_t value, size_t alignment)
{
	if (alignment == 0)
		ERROR("align_up called with zero alignment");
	if ((alignment & (alignment - 1)) != 0)
		ERROR("align_up currently requires power-of-two alignment");
	size_t remainder = value % alignment;
	if (remainder == 0)
		return value;
	return checked_add(value, alignment - remainder, "align_up overflow");
}

bool address_ranges_overlap(size_t offset_a, size_t size_a, size_t offset_b, size_t size_b)
{
	size_t end_a = checked_add(offset_a, size_a, "address range overflow");
	size_t end_b = checked_add(offset_b, size_b, "address range overflow");
	return offset_a < end_b && offset_b < end_a;
}

bool allocations_conflict(const ArenaAllocation& a, const ArenaAllocation& b)
{
	TensorLifetime life_a{a.tensor, a.first_use, a.last_use, a.size, a.alignment};
	TensorLifetime life_b{b.tensor, b.first_use, b.last_use, b.size, b.alignment};
	return lifetimes_overlap(life_a, life_b) && address_ranges_overlap(a.offset, a.size, b.offset, b.size);
}

bool validate_arena_plan(const ArenaPlan& plan, const std::vector<TensorLifetime>& lifetimes, std::string* reason)
{
	std::map<const Tensor*, TensorLifetime> lifetime_by_tensor;
	for (const TensorLifetime& lifetime : lifetimes)
		lifetime_by_tensor[lifetime.tensor] = lifetime;

	std::set<const Tensor*> seen;
	for (const ArenaAllocation& allocation : plan.allocations) {
		if (allocation.tensor == nullptr) {
			if (reason)
				*reason = "allocation without tensor";
			return false;
		}
		if (!seen.insert(allocation.tensor).second) {
			if (reason)
				*reason = "duplicate tensor allocation";
			return false;
		}
		auto life_it = lifetime_by_tensor.find(allocation.tensor);
		if (life_it == lifetime_by_tensor.end()) {
			if (reason)
				*reason = "allocation for unknown tensor";
			return false;
		}
		if (allocation.alignment == 0 || (allocation.alignment & (allocation.alignment - 1)) != 0) {
			if (reason)
				*reason = "invalid allocation alignment";
			return false;
		}
		if ((allocation.offset % allocation.alignment) != 0) {
			if (reason)
				*reason = "misaligned allocation offset";
			return false;
		}
		if (allocation.size != life_it->second.size_bytes || allocation.alignment != life_it->second.alignment || allocation.first_use != life_it->second.first_use || allocation.last_use != life_it->second.last_use) {
			if (reason)
				*reason = "allocation metadata does not match tensor lifetime";
			return false;
		}
		if (allocation.offset > plan.arena_size) {
			if (reason)
				*reason = "allocation offset beyond arena size";
			return false;
		}
		if (allocation.size > 0) {
			if (allocation.offset > SIZE_MAX - allocation.size) {
				if (reason)
					*reason = "allocation end overflow";
				return false;
			}
			if (allocation.offset + allocation.size > plan.arena_size) {
				if (reason)
					*reason = "allocation exceeds arena size";
				return false;
			}
		}
	}

	if (seen.size() != lifetimes.size()) {
		if (reason)
			*reason = "eligible tensor count does not match allocation count";
		return false;
	}

	for (size_t i = 0; i < plan.allocations.size(); i++) {
		for (size_t j = i + 1; j < plan.allocations.size(); j++) {
			if (allocations_conflict(plan.allocations[i], plan.allocations[j])) {
				if (reason)
					*reason = "simultaneously-live tensors overlap in memory";
				return false;
			}
		}
	}

	return true;
}

namespace {

bool less_name(const TensorLifetime& a, const TensorLifetime& b)
{
	return a.tensor->name < b.tensor->name;
}

ArenaPlan pack(const std::vector<TensorLifetime>& lifetimes)
{
	std::vector<TensorLifetime> sequence = lifetimes;
	std::sort(sequence.begin(), sequence.end(), [](const TensorLifetime& a, const TensorLifetime& b) {
		if (a.size_bytes != b.size_bytes) return a.size_bytes > b.size_bytes;
		return less_name(a, b);
	});

	ArenaPlan result;
	for (const TensorLifetime& life : sequence) {
		ArenaAllocation candidate{life.tensor, 0, life.size_bytes, life.alignment, life.first_use, life.last_use};
		size_t offset = 0;
		while (true) {
			offset = align_up(offset, life.alignment);
			candidate.offset = offset;
			bool conflict = false;
			for (const auto& placed : result.allocations) {
				if (allocations_conflict(candidate, placed)) {
					offset = checked_add(placed.offset, placed.size, "candidate offset overflow");
					conflict = true;
					break;
				}
			}
			if (!conflict) break;
		}
		result.allocations.push_back(candidate);
		result.arena_size = std::max(result.arena_size, checked_add(offset, life.size_bytes, "arena size overflow"));
	}
	return result;
}

} // namespace

arena_strategy parse_arena_strategy(const std::string& value)
{
	if (value == "first-fit") return arena_strategy::first_fit;
	if (value == "memory-schedule") return arena_strategy::memory_schedule;
	ERROR("bad command line argument for '--arena-strategy': " << value);
}

const char* arena_strategy_name(arena_strategy strategy)
{
	return strategy == arena_strategy::memory_schedule ? "memory-schedule" : "first-fit";
}

ArenaPlan TensorArenaPlanner::plan(const std::vector<TensorLifetime>& lifetimes) const
{
	return pack(lifetimes);
}

ArenaPlan TensorArenaPlanner::plan(const std::vector<TensorLifetime>& lifetimes, arena_strategy strategy, TensorArenaMetrics* metrics) const
{
	if (strategy != arena_strategy::first_fit && strategy != arena_strategy::memory_schedule)
		ERROR("arena planner only supports first-fit placement");
	ArenaPlan result = pack(lifetimes);
	if (metrics) {
		metrics->placement_strategy = "first-fit";
	}
	return result;
}

} // namespace toC
