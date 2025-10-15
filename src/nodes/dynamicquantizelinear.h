/* This file is part of onnx2c.
 *
 * DynamicQuantizeLinear node.
 * Quantizes a tensor from float to uint8_t.
 * Additional outputs are the scale and zeropoint
 * so that y = x / scale + zeropoint
 */ 
namespace toC {

class DynamicQuantizeLinear : public Node {
	public:
	DynamicQuantizeLinear() {
		op_name = "DynamicQuantizeLinear";
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			ERROR("DynamicQuantizeLinear should not have attributes, found" << a.name());
		}
	}


	virtual void print(std::ostream &dst) const override
	{
		const Tensor *x = get_input_tensor(0);
		int n_el = x->data_num_elem();

		INDT_1 << "/* DynamicQuantizeLinear */" << std::endl;

		INDT_1 << "float *in_data = (float*)x;" << std::endl;
		INDT_1 << "uint8_t *out_data = (uint8_t*)y;" << std::endl;
		INDT_1 << "float *y_scale_ = (float*)y_scale;" << std::endl;
		INDT_1 << "uint8_t *y_zero_point_ = (uint8_t*)y_zero_point;" << std::endl;
		INDT_1 << "float min, max; min=max=0.0;" << std::endl;

		INDT_1 << "for (int i=0; i<" << n_el << "; i++ ) {" << std::endl;
			INDT_2 << "float xi = in_data[i];" << std::endl;
			INDT_2 << "min = xi<min ? xi : min;" << std::endl;
			INDT_2 << "max = xi>max ? xi : max;" << std::endl;
		INDT_1 << "}" << std::endl;

		// TODO: assert output is uint8. Reading between the lines says this will change in the future
		INDT_1 << "*y_scale_ = (max-min)/255;" << std::endl;

		INDT_1 << "float fl_zero_point = (0 - min) / *y_scale;" << std::endl;
		// specs say:
		// y_zero_point = cast(round(saturate(itermediate_zero_point)))
		// Where saturates to output type limits (i.e. 0,255)
		// "rounding to nearest ties to even" I don't understand this:
		//  1.5 rounds to 2.0? and 2.5 rounds to 2.0 too?? is there such a mode?
		// round() should be good enough
		INDT_1 << "fl_zero_point = fl_zero_point < 0 ? 0 : fl_zero_point;" << std::endl;
		INDT_1 << "fl_zero_point = fl_zero_point > 255 ? 255 : fl_zero_point;" << std::endl;
		INDT_1 << "*y_zero_point_ = round(fl_zero_point);" << std::endl;


		INDT_1 << "for (int i=0; i<" << n_el << "; i++ ) {" << std::endl;
			INDT_2 << "float scaled = in_data[i]/ *y_scale + *y_zero_point;" << std::endl;
			INDT_2 << "scaled = scaled < 0 ? 0 : scaled;" << std::endl;
			INDT_2 << "scaled = scaled > 255 ? 255 : scaled;" << std::endl;
			INDT_2 << "out_data[i] = round(scaled);" << std::endl;
		INDT_1 << "}" << std::endl;
	}


	virtual void resolve(void) override
	{
		const Tensor *x = get_input_tensor(0);
		name_input(0, "x");

		Tensor *t = new Tensor;
		t->data_dim = x->data_dim;
		t->data_type = onnx::TensorProto_DataType_UINT8;
		register_output(t, "y");

		t = new Tensor;
		t->data_type = onnx::TensorProto_DataType_FLOAT;
		register_output(t, "y_scale");

		t = new Tensor;
		t->data_type = onnx::TensorProto_DataType_UINT8;
		register_output(t, "y_zero_point");
	}
};
}

