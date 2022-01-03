/* This file is part of onnx2c.
 *
 * Resize a N-dimensional tensor.
 * Implements down or upscaling an "image" by picking nearest pixel,
 * linear or cubic (not implemented here :) interpolation.
 */
namespace toC {

class Resize : public Node {
	public:
	Resize() {
		op_name = "Resize";
		X=roi=scales=sizes=Y=NULL;
		coordinate_transformation_mode = "half_pixel";
		cubic_coeff_a = -0.75;
		exclude_outside = 0;
		extrapolation_value = 0.0;
		mode = "nearest";
		nearest_mode = "round_prefer_floor";
	}
	/* Node attributes */
	std::string coordinate_transformation_mode;
	float cubic_coeff_a;
	int64_t exclude_outside;
	float extrapolation_value;
	std::string mode;
	std::string nearest_mode;

	// inputs
	const Tensor *X;
	// optional inputs
	const Tensor *roi;
	const Tensor *scales;
	const Tensor *sizes;
	// outputs
	const Tensor *Y;

	std::vector<float>dim_scales; // 'scales' value when calculating coordinate transforms

	/* Parse attributes, if this node has them. */
	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
			if( a.name() == "coordinate_transformation_mode" )
				coordinate_transformation_mode = parse_attribute_string(a);
			else if( a.name() == "cubic_coeff_a" )
				cubic_coeff_a = parse_attribute_float(a);
			else if( a.name() == "exclude_outside" )
				exclude_outside = parse_attribute_int(a);
			else if( a.name() == "extrapolation_value" )
				extrapolation_value = parse_attribute_float(a);
			else if( a.name() == "mode" )
				mode = parse_attribute_string(a);
			else if( a.name() == "nearest_mode" )
				nearest_mode = parse_attribute_string(a);
			else
				ERROR("Unknown attribute in node Resize");
		}
	}


	/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs) override
	{
		X = inputs[0];

		if (inputs.size() == 2)
			roi = inputs[1];
		if (inputs.size() == 3 && inputs[2]->name != "")
			scales = inputs[2];
		if (inputs.size() == 4)
			sizes = inputs[3];

		// "One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified."
		if (scales == NULL && sizes == NULL)
			ERROR("Resize node needs one of the optional input tensors 'scales' or 'sizes'");
		if (scales != NULL && sizes != NULL)
			ERROR("Resize node cannot have both optional input tensors 'scales' or 'sizes' given");

		if( sizes && sizes->isConst == false )
			ERROR("Unimplemented: Resize 'sizes' input is not a compile-time constant");


		std::vector<int64_t> output_size;
		if( sizes ) {
			for( int d=0; d<sizes->data_num_elem(); d++ ) {
				int64_t size = sizes->get_data_element(d);
				output_size.push_back(size);
				dim_scales.push_back( (float)size/X->data_dim[d]);
			}
		}
		else {
			for( int d=0; d<scales->data_num_elem(); d++ ) {
				float scale = scales->get_data_element_float(d);
				float size = scale * X->data_dim[d];
				output_size.push_back( floor(size) );
				dim_scales.push_back(scale);
			}
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

		/* TODO: optional outputs? */
	}


	/* Print the function parameters - use the order they are introduced in the
	 * ONNX documentation */
	virtual void print_parameters(std::ostream &dst, bool decorate ) const override
	{
		X->print_tensor_as_const(dst, !decorate);

		if (roi) {
			dst << ", ";
			roi->print_tensor_as_const(dst, !decorate);
		}
		if (scales) {
			dst << ", ";
			scales->print_tensor_as_const(dst, !decorate);
		}
		if (sizes) {
			dst << ", ";
			sizes->print_tensor_as_const(dst, !decorate);
		}

		dst << ", ";
		Y->print_tensor(dst, !decorate);
	}

	/* Print the coordinate transform algorithm, without integer roundings. */
	std::string coordinate_transformation( int dim, std::string y_coordinate) const
	{
		std::string scale = std::to_string(dim_scales[dim]);
		std::string x_dimsize = std::to_string(X->data_dim[dim]);
		std::string y_dimsize = std::to_string(Y->data_dim[dim]);

		std::string tf = "";
		if( coordinate_transformation_mode == "half_pixel" ) {
			// x_original = (x_resized + 0.5) / scale - 0.5
			// last "-0.5001" is a hack around rounding rules and floating point accuracies
			tf+= "(" + y_coordinate + " + 0.5" + ")/";
			tf+=     scale + "-0.500001";
		}
		else if (coordinate_transformation_mode == "tf_half_pixel_for_nn" ) {
			// These coordinate calculation algorithms are insanely labile.
			// I'm not at sure if the hack of 0.4999==0.5 is the correct one,
			// but it makes the backend tests pass
			tf+= "(" + y_coordinate + "+0.49999)/" + scale;
		}
		else if( coordinate_transformation_mode == "asymmetric" ) {
			// x_original = x_resized / scale
			tf+= y_coordinate + "/" + scale;
		}
		else if( coordinate_transformation_mode == "align_corners" ) {
			// x_original = x_resized * (length_original - 1) / (length_resized - 1)
			// this hack is not in the specs, but seems to make sense
			if( y_dimsize == "1" )
				tf+= "0";
			else
				tf+= y_coordinate + "*("+ x_dimsize + "-1)/(" + y_dimsize + "-1)";
		}
		else if( coordinate_transformation_mode == "pytorch_half_pixel" ) {
			// x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
			if( Y->data_dim[dim] > 1 )
				tf += "(" + y_coordinate + " +0.5)/" + scale + " - 0.5";
			else
				tf+= "0";
		}
		else
			ERROR("Resize: unimplemented coordinate_transformation_mode. Sorry. Patches welcome :)");

		return tf;
	}

	/* For the mode 'nearest', calculate the rounding of x_resized to indexes */
	std::string x_coord_nearest( int dim) const
	{
		std::string x_dimsize = std::to_string(X->data_dim[dim]);
		std::string x_resized = "x_orig_" + std::to_string(dim);
		// Apply rounding
		// floating point rounding is a mess.
		// ONNX defines 4 modes, 3 of which are implementable with C
		// The one missing is the case of "round 0.5 to 0".
		// These rounding modes should not matter too much when calculating
		// neural networks, but here the rounding specifies distinct input
		// pixels, AND someone thought they would be needing this exact
		// rounding form. So much that it is the default.
		std::string roundf;
		if( nearest_mode == "round_prefer_floor" ) {
			LOG(WARNING) << "The selected nearest_mode rounding implementation is not fully accurate" << std::endl;
			roundf = "roundf";
		}
		else if( nearest_mode == "round_prefer_ceil" )
			roundf = "roundf";
		else if( nearest_mode == "floor")
			roundf = "floor";
		else if( nearest_mode == "ceil")
			roundf = "ceil";
		else
			ERROR("Unkown nearest_mode");
		roundf += "(" + x_resized + ")";

		// Bound the result index to within the dimension size. This is implicitly expected in the specs.
		std::string rv = "MIN(" + roundf + ", " + x_dimsize + " -1)";
		return rv;
	}

	void print_calc_nearest(std::ostream &dst) const
	{
		std::string out = "Y";
		std::string in = "X";
		unsigned n_data_dims = Y->rank();
		for( unsigned i = 0; i<n_data_dims; i++) {
			INDT_2 << "uint32_t x"<<i << " = " << x_coord_nearest( i ) << ";" << std::endl;
			out += "[o" + std::to_string(i) + "]";
			in += "[x" + std::to_string(i) + "]";
		}
		INDT_2 << out << " = " << in << ";" << std::endl;
	}
	
	void print_calc_linear(std::ostream &dst) const
	{
		std::string out = "Y";
		std::string in = "X";
		std::vector<int> interpolate_dims;
		unsigned n_data_dims = Y->rank();
		unsigned resized_dims = 0;
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			// TODO: or if Xi == Yi. No interpolation
			out+="[o" + i_str + "]";
			if( X->data_dim[i] == 1 ){
				continue;
			}
			interpolate_dims.push_back(i);

			resized_dims++;
			if( resized_dims > 2 )
				ERROR("Resize over more than 2 dimensions is not implemented");
		}


		if( interpolate_dims.size() == 1 )
		{

			for( unsigned i = 0; i<n_data_dims; i++) {
				if( X->data_dim[i] == 1 ) {
					in+="[0]";
					continue;
				}

				INDT_2 << "unsigned a = floor(x_orig_" << i << ");" << std::endl;
				INDT_2 << "unsigned b = MIN(ceil(x_orig_" << i << "), ";
				   dst << X->data_dim[i] << "-1);" << std::endl;
				INDT_2 << "float w = x_orig_" << i << "-a;" << std::endl;
				in += "[c]";
			}
			INDT_2 << "unsigned c;" << std::endl;
			INDT_2 << "c=a; float A="<<in<<";"<<std::endl;
			INDT_2 << "c=b; float B="<<in<<";"<<std::endl;

			INDT_2 << out << " = A*(1-w) + B*w;" << std::endl;
		}

		// TODO: this most likely could be generalized to N-D interpolation
		else if( interpolate_dims.size() == 2 )
		{
			bool second_dim = false;
			for( unsigned i = 0; i<n_data_dims; i++) {
				if( X->data_dim[i] == 1 ) {
					in+="[0]";
					continue;
				}

				// The "inner dimension" of the bilinear interpolation
				if( second_dim )
				{
					INDT_2 << "unsigned y1 = MAX(floor(x_orig_" << i << "), 0);" << std::endl;
					INDT_2 << "unsigned y2 = MIN(ceil(x_orig_" << i << "), ";
					   dst << X->data_dim[i] << "-1);" << std::endl;
					INDT_2 << "float w_y = x_orig_" << i << "-y1;" << std::endl;
					in += "[y]";
				}

				// The "outer dimension" of the biliniear interpolation
				else {
					INDT_2 << "unsigned x1 = MAX(floor(x_orig_" << i << "), 0);" << std::endl;
					INDT_2 << "unsigned x2 = MIN(ceil(x_orig_" << i << "), ";
					   dst << X->data_dim[i] << "-1);" << std::endl;
					INDT_2 << "float w_x = x_orig_" << i << "-x1;" << std::endl;
					in += "[x]";
				}
				second_dim = true;
			}
			// Figure with these names here: https://chao-ji.github.io/jekyll/update/2018/07/19/BilinearResize.html
			INDT_2 << "unsigned x,y;" << std::endl;
			INDT_2 << "x=x1; y=y1; float A="<<in<<";"<<std::endl;
			INDT_2 << "x=x2; y=y1; float B="<<in<<";"<<std::endl;
			INDT_2 << "x=x1; y=y2; float C="<<in<<";"<<std::endl;
			INDT_2 << "x=x2; y=y2; float D="<<in<<";"<<std::endl;
			INDT_2 << "float AB = A*(1-w_x) + B*w_x;" << std::endl;
			INDT_2 << "float CD = C*(1-w_x) + D*w_x;" << std::endl;
			INDT_2 << "float ABCD = AB*(1-w_y) + CD*w_y;" << std::endl;
			INDT_2 << out << " = ABCD;" << std::endl;
		}

		else
			ERROR("Resize. Only 1D and 2D interpolation implemented");

	}
	
	void print_calc_output(std::ostream &dst) const
	{
		if ( mode == "nearest" )
			print_calc_nearest(dst);
		else if( mode == "linear" )
			print_calc_linear(dst);
		else if( mode == "cubic" )
			ERROR("Unimplemented: cubic interpolation in resize");
		else
			ERROR("Unknown interpolation mode");
	}

	/* Body of the node implementing function */
	virtual void print(std::ostream &dst) const override
	{

		INDT_1 << "/* Resize" << std::endl;
		INDT_1 << " * attributes: " << std::endl;
		INDT_1 << " * coordinate_transformation_mode: " << coordinate_transformation_mode << std::endl;
		INDT_1 << " * cubic_coeff_a: " << cubic_coeff_a << std::endl;
		INDT_1 << " * exclude_outside: " << exclude_outside << std::endl;
		INDT_1 << " * extrapolation_value: " << extrapolation_value << std::endl;
		INDT_1 << " * mode: " << mode << std::endl;
		INDT_1 << " * nearest_mode: " << nearest_mode << std::endl;
		INDT_1 << " *" << std::endl;
		INDT_1 << " * dimension scaling factors:" << std::endl;
		for( auto s : dim_scales )
			INDT_1 << " * " << s << std::endl;
		INDT_1 << " */" << std::endl;

		unsigned n_data_dims = Y->rank();

		INDT_1 << cast_to_ndim_arrayptr(X, "X") << std::endl;
		INDT_1 << cast_to_ndim_arrayptr(Y, "Y") << std::endl;

		// loop over output
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			std::string o_idx = "o" + i_str;
			INDT_1 << "for( uint32_t " << o_idx << "=0; ";
			   dst <<       o_idx << "<" << Y->data_dim[i] << "; ";
			   dst <<       o_idx <<"++) {" << std::endl;

			INDT_2 << "float x_orig_" << i_str << " = (float)" << coordinate_transformation( i, o_idx ) <<";" << std::endl;
		}

		print_calc_output(dst);

		// close the loops over output
		for( unsigned i = 0; i<n_data_dims; i++) {
			INDT_1 << "}" << std::endl;
		}
	}
};
}

