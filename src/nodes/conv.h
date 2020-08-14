
namespace toC {

class Conv : public Node {
	public:
	Conv() {
		op_name = "Conv";
		auto_pad = "NOTSET";
		group = 1;
	}
	/* Conv node specific attributes */
	std::string auto_pad;
	std::vector<int> dilations;
	int group;
	std::vector<int> kernel_shape;
	std::vector<int> pads;
	std::vector<int> strides;

	void parseAttributes_auto_pad( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_STRING )
			ERROR("Wrong attribute type for Conv attribute 'auto_pad'");

		auto_pad = a.s();

		if( a.s() == "SAME_UPPER" )
			; // all good :)
		else if( a.s() == "SAME_LOWER" ) {
			ERROR("Unimplemented: SAME_LOWER padding for Conv");
		}
		else if( a.s() == "VALID" ) {
			ERROR("Unimplemented: VALID padding for Conv");
		}
		else if (a.s() == "NOTSET" ) {
			;
		}
		else
			ERROR("Unknown Conv Attribute auto_pad = " << a.name());
	}

	void parseAttributes_dilations( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for Conv attribute 'dilations'");

		for( auto i : a.ints() ) {
			dilations.push_back(i);
		}
	}

	void parseAttributes_group( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INT )
			ERROR("Wrong attribute type for Conv attribute 'group'");
		group = a.i();
	}

	void parseAttributes_kernel_shape( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for Conv attribute 'kernel_shape'");

		for( auto i : a.ints() ) {
			kernel_shape.push_back(i);
		}
	}

	void parseAttributes_pads( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for Conv attribute 'pads'");

		for( auto i : a.ints() ) {
			pads.push_back(i);
		}
	}

	void parseAttributes_strides( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for Conv attribute 'strides'");

		for( auto i : a.ints() ) {
			strides.push_back(i);
		}
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto a : node.attribute() ) {
			if( a.name() == "auto_pad" )
				parseAttributes_auto_pad(a);
			else if( a.name() == "dilations" )
				parseAttributes_dilations(a);
			else if( a.name() == "group" )
				parseAttributes_group(a);
			else if( a.name() == "kernel_shape" )
				parseAttributes_kernel_shape(a);
			else if( a.name() == "pads" )
				parseAttributes_pads(a);
			else if( a.name() == "strides" )
				parseAttributes_strides(a);
		}
	}

	virtual void print(std::ostream &dst) const
	{
		const Tensor *x = inputs[0];
		const Tensor *w = inputs[1];
		const Tensor *y = outputs[0];
		std::string type = x->data_type_str();
		int num_data_dim = x->data_dim.size()-2;

		/* Print settings into comment section before code */
		dst << "\t/* Conv" << std::endl;
		dst << "\t *" << std::endl;
		dst << "\t * auto_pad: " << auto_pad <<std::endl;
		dst << "\t * dilations: ";
			for( int d: dilations )
				dst << d << " ";
		dst << std::endl;
		dst << "\t * group: " << group <<std::endl;
		dst << "\t * kernel_shape: ";
			for( int k: kernel_shape )
				dst << k << " ";
		dst << std::endl;
		dst << "\t * pads: ";
			for( int p: pads)
				dst << p << " ";
		dst << std::endl;
		dst << "\t * strides: ";
			for( int s: strides)
				dst << s << " ";
		dst << std::endl <<  "\t */" << std::endl;

		/* TODO: select between alternative implementations on how to pad:
		 * -scratch area larger matrix with padding built-in
		 * -special case the borders outside of the inner loop
		 * -special case the borders inside of the inner loop
		 * For now, hard-code scratch-area.
		 */

		dst << "\t/* Loop over batches */" << std::endl;
		dst << "\tfor( uint32_t b=0; b<" << x->data_dim[0] << "; b++) {" << std::endl << std::endl;

		/* Create scratch-area with paddings. Size is [channels][dim1+pads][dim2+pads]*/
		/* TODO: don't put this on stack! Not putting it on stack needs rework in Graph::print* functions,
		 * so it is here only to speed up onnx2c development. */
		std::vector<int> scr_s;
		scr_s.push_back(x->data_dim[1]);
		scr_s.push_back(x->data_dim[2] + pads[0] + pads[0+num_data_dim]);
		scr_s.push_back(x->data_dim[3] + pads[1] + pads[1+num_data_dim]);

		dst << "\t/* Copy over input to scratch pad memory */" << std::endl;
		dst << "\t"       << type << " scratch[" << scr_s[0] << "]["<< scr_s[1];
		dst <<               "]["<< scr_s[2] << "];" << std::endl;

		dst << "\t"       << "for( uint32_t c=0; c<" << x->data_dim[1] <<"; c++) {" << std::endl;

		dst << "\t\t"     << "for( uint32_t i1=" << pads[0] << ";";
		dst <<                   "i1<" << scr_s[1]-pads[0+num_data_dim] << ";";
		dst <<                   "i1++ ) {" << std::endl;
		dst << "\t\t\t"   << "for( uint32_t i2=" << pads[1] << ";";
		dst <<                   "i2<" << scr_s[2]-pads[1+num_data_dim] << ";";
		dst <<                   " i2++ ) {" << std::endl;

		/* TODO: batch size is fixed to 1 here */
		dst << "\t\t\t\t" << "scratch[c][i1][i2] = " << x->cname() << "[b][c][i1-"<<pads[0]<<"][i2-"<<pads[1]<<"];" << std::endl;
		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;
		dst << "\t}"      << std::endl;

		dst << "\t/* Pad the scratch pad memory */" << std::endl;
		dst << "\t"       << "for( uint32_t c=0; c<" << x->data_dim[1] <<"; c++) {" << std::endl;

		dst << "\t\t"     << "/*'Top' of 1st dimension */" << std::endl;
		dst << "\t\t"     << "for( uint32_t i1=0; i1<" << pads[0] << "; i1++) {" << std::endl;
		dst << "\t\t\t"   << "for( uint32_t i2=" << pads[1] << "; ";
		dst <<                   "i2<" << scr_s[2]-pads[1+num_data_dim] << ";";
		dst <<                   "i2++ ) {" << std::endl;

		dst << "\t\t\t\t" << "scratch[c][i1][i2] = ";
		if( auto_pad == "NOTSET" )
			dst << "0;" << std::endl;
		else // SAME_*
			dst << x->cname() << "[b][c]["<< pads[0] << "][i2-"<<pads[1]<<"];" << std::endl;

		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;

		dst << "\t\t"     << "/*'Bottom' of 1st dimension */" << std::endl;
		dst << "\t\t"     << "for( uint32_t i1="<<scr_s[1]-pads[0+num_data_dim]<< "; i1<" << scr_s[1] << "; i1++) {" << std::endl;
		dst << "\t\t\t"   << "for( uint32_t i2=" << pads[1] << "; ";
		dst <<                   "i2<" << scr_s[2]-pads[1+num_data_dim] << ";";
		dst <<                   "i2++ ) {" << std::endl;

		dst << "\t\t\t\t" << "scratch[c][i1][i2] = ";
		if( auto_pad == "NOTSET" )
			dst << "0;" << std::endl;
		else // SAME_*
			dst << x->cname() << "[b][c]["<< x->data_dim[3]-1 << "][i2-"<<pads[1]<<"];" << std::endl;


		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;


		dst << "\t\t"     << "/*'Top' of 2nd dimension */" << std::endl;
		dst << "\t\t"     << "for( uint32_t i1=0; i1<" << scr_s[1] << "; i1++) {" << std::endl;
		dst << "\t\t\t"   << "for( uint32_t i2=0; i2<" << pads[1] << "; i2++) {" << std::endl;

		dst << "\t\t\t\t" << "scratch[c][i1][i2] = ";
		if( auto_pad == "NOTSET" )
			dst << "0;" << std::endl;
		else // SAME_*
			dst << "scratch[c][i1]["<< pads[1] <<"];" << std::endl;
		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;

		dst << "\t\t"     << "/*'Bottom' of 2nd dimension */" << std::endl;
		dst << "\t\t"     << "for( uint32_t i1=0; i1<" << scr_s[1] << "; i1++) {" << std::endl;
		dst << "\t\t\t"   << "for( uint32_t i2=" << scr_s[2] - pads[1+num_data_dim] << "; ";
		dst <<                   "i2<" << scr_s[2] << ";";
		dst <<                   "i2++ ) {" << std::endl;

		dst << "\t\t\t\t" << "scratch[c][i1][i2] = ";
		if( auto_pad == "NOTSET" )
			dst << "0;" << std::endl;
		else // SAME_*
			dst << "scratch[c][i1]["<< scr_s[2]-pads[1]-1 <<"];" << std::endl;


		dst << "\t\t\t}"  << std::endl;
		dst << "\t\t}"    << std::endl;
		dst << "\t}"      << std::endl;


		std::string out = y->cname();
		dst << "\t/* Run the convolution */" << std::endl;
		dst << "\t/* loop over: m=input maps, c=channels, i1&i2 data dimensions*/" << std::endl;
		dst << "\tfor( uint32_t m=0; m<" << w->data_dim[0] << "; m++) {" << std::endl;
		dst << "\tfor( uint32_t i1=0, o1=0; ";
		dst <<        "i1<" << scr_s[1] << "; ";
		dst <<        "i1+=" << strides[0] << ", o1++) {" << std::endl;
		dst << "\tfor( uint32_t i2=0, o2=0; ";
		dst <<        "i2<" << scr_s[2] << "; ";
		dst <<        "i2+=" << strides[1] <<", o2++) {" << std::endl;

		/* Loop over the kernel */
		dst << "\t\t" << out << "[b][m][o1][o2] = 0;" << std::endl;;
		dst << "\t\tfor( uint32_t c=0; c<" << x->data_dim[1] << "; c++) {" << std::endl;
		dst << "\t\tfor( uint32_t k1=0; k1<" << kernel_shape[0] << "; k1++) {" << std::endl;
		dst << "\t\tfor( uint32_t k2=0; k2<" << kernel_shape[0] << "; k2++) {" << std::endl;

		dst << "\t\t\t" << out << "[b][m][o1][o2] += scratch[c][i1+k1][i2+k2] *";
		dst <<             w->cname() << "[m][c][k1][k2];" << std::endl;
			
		dst << "\t\t}"      << std::endl;
		dst << "\t\t}"      << std::endl;
		dst << "\t\t}"      << std::endl;
		
		dst << "\t}"      << std::endl;
		dst << "\t}"      << std::endl;
		dst << "\t}"      << std::endl;

		dst << "\t} /* batch */"      << std::endl;
	}
 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		const Tensor *x = inputs[0]; // data
		const Tensor *w = inputs[1]; // weights
		const Tensor *b;             // bias
		if( inputs.size() == 3 ) {
			b = inputs[2];
			ERROR("unimplemented: bias term");
		}
		else
			b = NULL;

		if(  typeConstraint_highPrecisionNumeric(x) == false
		   ||typeConstraint_highPrecisionNumeric(w) == false)
			ERROR("Incorrect input for node");
		if( b && (typeConstraint_highPrecisionNumeric(b) == false) )
			ERROR("Incorrect input for node");

		if( x->data_dim.size() != 4 )
			ERROR("Unimplemented: Conv for non 2D images");

		int num_data_dim = x->data_dim.size()-2;


		/* TODO: assume here "pads" attribute is not given, and auto_pad iseither of SAME_*.
		 *       also assume kernel size is odd (not even).
		 * TODO: if auto_pad is VALID, does it imply kernel size is 1, so no padding is ever
		 *       required, or does the output shrink? ONNX spec just assumes this as known...
		 */
		if( kernel_shape[0] != kernel_shape[1] )
			ERROR("Unimplemented: non-square kernels");

		if( (kernel_shape[0] & 0x1) == 0 )
			ERROR("Unimplemented: even (not odd) sized kernels");

		if( pads.size() == 0 ) {
			for( unsigned i=0; i< x->data_dim.size() -2; i++ ) {
				pads.push_back( kernel_shape[0]/2 );
				pads.push_back( kernel_shape[0]/2 );
			}
		}

		if( group != 1 )
			ERROR("Unimplemented: Conv: setting group to anything but 1");

		for( int d : dilations )
			if( d != 1 )
				ERROR("Unimplemented: Conv: dilations other than 1");

		Tensor *rv = new Tensor;
		rv->data_dim.push_back(x->data_dim[0]);//batch size
		rv->data_dim.push_back(w->data_dim[0]);//"number of feature maps"

		for( unsigned xdim=2; xdim < x->data_dim.size(); xdim++) {
			int outdim;
			unsigned dim = xdim-2;
			// From ONNX Operators.md:
			// SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.
			// "match" here means "is equal".
			if( auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" )
				outdim = x->data_dim[xdim];
			else if( auto_pad == "NOTSET" ) {
				//padded input
				int input_size = x->data_dim[xdim] + pads[dim]+pads[dim+num_data_dim];
				// [ 0 1 2 3 4 5 6 7 8 9  ]
				//                |kern=3|
				// last output=7
				int last_out = input_size - kernel_shape[dim];
				outdim = last_out / strides[dim] + 1;
			}
			else
				ERROR("Unimplemented: VALID padding");

			rv->data_dim.push_back(outdim);
		}

		rv->data_type = x->data_type;
		rv->data_num_elem = x->data_num_elem;
		outputs.push_back(rv);
	}
};
}
