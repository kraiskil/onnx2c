/* This file is part of onnx2c.
 *
 * Upsample a (image) tensor.
 * This operator has been deprecated in ONNX opset 10,
 * seemingly being replaced with the Resize operator.
 */
#include "resize.h"
namespace toC {

class Upsample : public Resize {
	public:
	Upsample() {
		op_name = "Upsample";
	}

	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		X = inputs[0];

		scales = inputs[1];

		if( scales->isConst == false )
			ERROR("Unimplemented: Upsample 'sizes' input is not a compile-time constant: " + scales->name);


		std::vector<int64_t> output_size;
		for( int d=0; d<scales->data_num_elem(); d++ ) {
			float scale = scales->get_data_element_float(d);
			float size = scale * X->data_dim[d];
			output_size.push_back( floor(size) );
			dim_scales.push_back(scale);
		}

		/* Create output tensors.
		 * Set data dimensions and data type for the created tensors. */
		Tensor *t = new Tensor;


		for( auto s : output_size )
			t->data_dim.push_back(s);
		t->data_type = onnx::TensorProto_DataType_FLOAT;
		/* Store the created tensor both as reference in this node, and into
		 * the return value vector! */
		Y = t;
		outputs.push_back(t);
	}


};
}

