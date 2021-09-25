/* This file is part of onnx2c.
 *
 * Common parts of spatial filters.
 * This is a parent class for different
 * convolution and pooling nodes.
 * This class is not intended to create
 * node objects from, only to contain
 * shared code.
 */

#pragma once
namespace toC {

class SpatialFilter : public Node {

	public:
	SpatialFilter() {
		auto_pad = "NOTSET";
		x=w=y=NULL;
		group=1;
	}
	// inputs
	const Tensor *x;
	const Tensor *w;
	// outputs
	const Tensor *y;

	// Attributes
	std::vector<int> kernel_shape;
	std::string auto_pad;
	std::vector<int> dilations;
	int group;
	std::vector<int> pads;
	std::vector<int> strides;


	virtual void parseAttributes( onnx::NodeProto &node ) override {
		for( const auto& a : node.attribute() ) {
			if( a.name() == "auto_pad" )
				auto_pad = parse_attribute_string(a);
			else if( a.name() == "dilations" )
				dilations = parse_attribute_ints(a);
			else if( a.name() == "group" )
				group = parse_attribute_int(a);
			else if( a.name() == "kernel_shape" )
				kernel_shape = parse_attribute_ints(a);
			else if( a.name() == "pads" )
				pads = parse_attribute_ints(a);
			else if( a.name() == "strides" )
				strides = parse_attribute_ints(a);
		}
	}


	void resolve_strides(void)
	{
		unsigned num_data_dim = x->rank()-2;
		if( strides.size() == 0 )
			for( unsigned i=0; i<num_data_dim; i++)
				strides.push_back(1);
	}

	void resolve_kernel_shape(void)
	{
		//if kernel shape is not given, infer from w
		if( kernel_shape.size() == 0 ) {
			// skip M and C/group dimensions
			for( unsigned i=2; i<w->rank(); i++)
				kernel_shape.push_back(w->data_dim[i]);
		}
	}

	void resolve_dilations(void)
	{
		unsigned num_data_dim = x->rank()-2;
		if( dilations.size() == 0 )
			for( unsigned i=0; i< num_data_dim; i++ )
				dilations.push_back(1);
	}
	void resolve_pads(void)
	{
		unsigned num_data_dim = x->rank()-2;
		if( pads.size() == 0 ) {
			pads.resize(num_data_dim*2);
			for( unsigned i=0; i< num_data_dim; i++ ) {
				if( auto_pad == "VALID" || auto_pad == "NOTSET" ) {
					pads[i] = 0;
					pads[i+num_data_dim] = 0;
				}
				else {
					// TODO: diations and strides might cause need for bigger paddings
					pads[i] = kernel_shape[i] / 2;
					pads[i+num_data_dim] = kernel_shape[i] / 2;
					// TODO: handle case where uneven padding is needed
				}
			}
		}
	}

	virtual std::vector<int> resolve_output_size(void)
	{
		std::vector<int> rv;
		unsigned num_data_dim = x->rank()-2;
		rv.push_back(x->data_dim[0]);//batch size
		rv.push_back(w->data_dim[0]);//"number of feature maps"

		for( unsigned xdim=2; xdim < x->data_dim.size(); xdim++) {
			int outdim;
			unsigned dim = xdim-2;
			// From ONNX Operators.md:
			// SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.
			// "match" here means "is equal".
			if( auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" )
				outdim = x->data_dim[xdim];
			else if( auto_pad == "NOTSET" || auto_pad == "VALID") {
				//padded input
				int input_size = x->data_dim[xdim] + pads[dim]+pads[dim+num_data_dim];
				// [ 0 1 2 3 4 5 6 7 8 9  ]
				//                |kern=3|
				// last output=7
				int last_out = input_size - kernel_shape[dim];
				outdim = last_out / strides[dim] + 1;
			}

			rv.push_back(outdim);
		}

		return rv;
	}


	void print_header_info_comment(std::ostream &dst) const
	{
		INDT_1 << "/* " << op_name << std::endl;
		INDT_1 << " *" << std::endl;
		INDT_1 << " * auto_pad: " << auto_pad <<std::endl;
		INDT_1 << " * dilations: ";
			for( int d: dilations )
				dst << d << " ";
			dst << std::endl;
		INDT_1 << " * group: " << group <<std::endl;
		INDT_1 << " * kernel_shape: ";
			for( int k: kernel_shape )
				dst << k << " ";
			dst << std::endl;
		INDT_1 << " * pads: ";
			for( int p: pads)
				dst << p << " ";
			dst << std::endl;
		INDT_1 << " * strides: ";
			for( int s: strides)
				dst << s << " ";
			dst << std::endl;
		INDT_1 << " */" << std::endl;
	}


	/* Print the loops of the convolution.
	 * This version has checks in the innermost loop for checking when
	 * the kernel hits paddings.
	 * This (probably, unless compliers are getting *real* amazing) causes
	 * a lot of overhead. A.k.a. optmization opportunities.
	 *
	 * Three callbacks to pure virtual functions are used:
	 * - to initialize output cell
	 * - to calculate input cell / kernel cell (this is the calculation in the innermost loop)
	 * - to finalize the output cell
	 */
	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx="") const = 0;
	virtual void print_output_cell_calc(std::ostream &dst, const std::string &x_idx="", const std::string &w_idx="", const std::string &y_idx="") const = 0;
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx="") const = 0;
	void print_loop_with_padding_checks(std::ostream &dst) const
	{
		unsigned n_data_dims = x->data_dim.size() -2;
		unsigned batch_size = x->data_dim[0];
		unsigned channels = x->data_dim[1];
		unsigned maps=0;
		if( w )
			maps = w->data_dim[0];

		/* Accumulate the size of the dimensions so that the Indices value
		 * can be calculated directly out of the indivicual indices.
		 * See ONNX specification on what the Indices output is */
		std::vector<int> size_of_dim(x->data_dim.size());
		size_of_dim[x->data_dim.size()-1]=1;
		for( int i=x->data_dim.size()-2; i>= 0; i--)
			size_of_dim[i] = size_of_dim[i+1] * x->data_dim[i];

		/* Create various indexing strings. This makes generating the loops much cleaner,
		 * and makes possible the code sharing in child classes. */
		std::string x_idx = "[b][c]";
		std::string in_kern_idxs = "[b][c]";
		std::string y_idx = "[b][c]";
		std::string indices_value = "(b*" + std::to_string(size_of_dim[0]) + ")+(c*" + std::to_string(size_of_dim[1]) + ")";
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			x_idx += "[i" + i_str + "]";
			y_idx += "[o" + i_str + "]";
			in_kern_idxs += "[ii" + i_str + "]";
			indices_value += "+(ii" + i_str + "*" + std::to_string(size_of_dim[i+2]) + ")";
		}

		/* Create the loops over batches and channels.
		 * In case this SpatialFilter has a weights input (w), this first loop is over
		 * output channels (M). Othervise input channels==outputchannels, and it is named C
		 */
		INDT_1 << "for( uint32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if( options.quantize ) {
			INDT_2 << "int32_t batch_min = INT32_MAX;" << std::endl;
			INDT_2 << "int32_t batch_max = INT32_MIN;" << std::endl;
		}
		if( w && group==1)
			INDT_1 << "for( uint32_t m=0; m<" << w->data_dim[0] << "; m++) {" << std::endl;
		else if( w && group > 1 ) {
			INDT_1 << "uint32_t go = " << maps/group     << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "uint32_t gi = " << channels/group << "; // inptput group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( uint32_t g=0; g<" << group << "; g++) {" << std::endl;
			INDT_1 << "for( uint32_t m=go*g; m<go*(g+1); m++) {" << std::endl;
		}
		else
			INDT_1 << "for( uint32_t c=0; c<" << x->data_dim[1] << "; c++) {" << std::endl;


		// loop over outputs and inputs
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string o_idx = "o" + std::to_string(i);
			std::string i_idx = "i" + std::to_string(i);
			INDT_2 << "for( int32_t " << o_idx << "=0, ";
			   dst <<       i_idx << "=" << -pads[i] << "; ";
			   dst <<       o_idx << "<" << y->data_dim[2+i] << "; ";
			   dst <<       o_idx <<"++, "<< i_idx << "+=" << strides[i] << ") {" << std::endl;
		}

		print_output_cell_init(dst, y_idx);

		if( w && group == 1 )
			INDT_3 <<   "for( int32_t c=0; c<" << channels << "; c++ ) {" << std::endl;
		else if( w && group > 1 )
			INDT_3 <<   "for( int32_t c=gi*g; c<gi*(g+1); c++ ) {" << std::endl;

		// loop over channels and kernel
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string idx = "k" + std::to_string(i);
			INDT_3 << "for( uint32_t " << idx << "=0; ";
			   dst <<       idx << "<" << kernel_shape[i] << "; ";
			   dst <<       idx <<"++ ) {" << std::endl;
		}

		// check for out-of-input reading (i.e. read a pad)
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			INDT_4 <<  "int ii" << i_str << " = i" << i_str << "+k" << i_str <<" + " << dilations[i] <<"-1;"<<std::endl;
			INDT_4 <<  "if( ii" << i_str << "<0) continue;" << std::endl;
			INDT_4 <<  "if( ii" << i_str << ">=" << x->data_dim[2+i] << ") continue;" << std::endl;
		}

		print_output_cell_calc(dst, in_kern_idxs, "", y_idx);

		// close kernel loop
		for( unsigned i = 0; i<n_data_dims; i++)
			INDT_3 << "} /* k */" << std::endl;

		// close input channels loop when it is separate from output channels
		if( w )
			INDT_3 << "} /* c */" << std::endl;
		print_output_cell_finalize(dst, y_idx);

		// close output loop
		for( unsigned i = 0; i<n_data_dims; i++)
			INDT_2 << "} /* o */" << std::endl;

		// close loops over batches and output channels
		INDT_1 << "} /* m or c, depending on this node's operator */" << std::endl;
		if( w && group > 1 )
			INDT_1 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}
};
}

