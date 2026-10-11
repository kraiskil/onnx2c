#pragma once

#include "../node.h"

namespace toC {

class Im2Col : public Node {
	public:
	enum ArithmeticMode {
		Conv,
		ConvInteger,
		QLinearConv,
	};
	enum Implementation {
		Implicit,
		Explicit,
	};

	Im2Col();
	std::string c_name(void) const;
	std::string c_output_decl(void);
	void print_node(std::ostream& dst) const;
	void print_implicit_node(std::ostream& dst) const;
	void print_explicit_node(std::ostream& dst) const;
	void resolve_datatypes(void);
	void print(std::ostream& destination) const override;
	void print_accumulator_init(std::ostream& dst) const;
	void print_accumulator_calc(std::ostream& dst, const std::string& x_idx, const std::string& w_idx) const;
	void print_accumulator_finalize(std::ostream& dst) const;

	ArithmeticMode arithmetic_mode = Conv;
	Implementation implementation = Implicit;
	int64_t batch = 1;
	int64_t in_ch = 1;
	int64_t out_ch = 1;
	int64_t in_h = 1;
	int64_t in_w = 1;
	int64_t kernel_h = 1;
	int64_t kernel_w = 1;
	int64_t stride_h = 1;
	int64_t stride_w = 1;
	int64_t pad_h = 0;
	int64_t pad_w = 0;
	int64_t dilation_h = 1;
	int64_t dilation_w = 1;
	int64_t out_h = 1;
	int64_t out_w = 1;
	int64_t group = 1;
	bool has_bias = false;
	bool has_x_zero_point = false;
	bool has_w_zero_point = false;
};

} // namespace toC
