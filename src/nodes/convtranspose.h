/* This file is part of onnx2c.
 *
 * Declarations for the ConvTranspose node.
 */

#include "error.h"
#include "onnx.pb.h"
#include "options.h"

#include "aixlog.hpp"
#include <iostream>
#include "spatialfilter.h"
namespace toC {

class ConvTranspose : public Node {
	public:
	ConvTranspose() {
		op_name = "ConvTranspose";
		x=w=y=b = NULL;
		output_shape_given = false;
		auto_pad = "NOTSET";
		group=1;
	}
	// inputs
	const Tensor *x;
	const Tensor *w;
	// optional inputs
	const Tensor *b;
	// outputs
	const Tensor *y;

	// Attributes
	std::vector<int64_t> kernel_shape;
	std::string auto_pad;
	std::vector<int64_t> dilations;
	int group;
	std::vector<int64_t> pads;
	std::vector<int64_t> strides;
	std::vector<int64_t> output_padding;
	std::vector<int64_t> output_shape;

	bool output_shape_given; // [sic] - should be output_shape_given

	virtual void parseAttributes( onnx::NodeProto &node ) override;

	virtual void resolve(void) override;
	std::vector<int> calculate_output_size(void);
	void resolve_strides(void);
	void resolve_kernel_shape(void);
	void resolve_dilations(void);
	void resolve_convtranspose_pads(void);
	void resolve_output_shape(void);
	void resolve_output_pads(void);

	virtual void print(std::ostream &dst) const override;
	void print_header_info_comment(std::ostream &dst) const;
	void print_calculation(std::ostream &dst) const;
};

} // namespace

