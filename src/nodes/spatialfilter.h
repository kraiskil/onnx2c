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
#include "node.h"
namespace toC {

class SpatialFilter : public Node {

	public:
	SpatialFilter() {
		auto_pad = "NOTSET";
		group=1;
	}

	// Attributes
	std::vector<int64_t> kernel_shape;
	std::string auto_pad;
	std::vector<int64_t> dilations;
	int group;
	std::vector<int64_t> pads;
	std::vector<int64_t> strides;

	const Tensor* get_X(void) const { return get_input_tensor(0); }
	const Tensor* get_W(void) const {
		if( get_number_of_inputs() > 1 )
			return get_input_tensor(1);
		else
			return nullptr;
	}
	const Tensor* get_Y(void) const { return get_output_tensor(0); }
	uint32_t get_numDataDim(void) const {return get_X()->rank() - 2; }

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
		if( strides.size() == 0 )
			for( unsigned i=0; i<get_numDataDim(); i++)
				strides.push_back(1);
		if( get_numDataDim() != strides.size() )
			ERROR("Dimension of the stride do not match data dimensions");
		for( uint64_t s : strides )
			if( s == 0 )
				ERROR("Stride of 0");
	}

	void resolve_kernel_shape(void)
	{
		//if kernel shape is not given, infer from w
		if( kernel_shape.size() == 0 ) {
			// skip M and C/group dimensions
			for( unsigned i=2; i<get_W()->rank(); i++)
				kernel_shape.push_back(get_W()->data_dim[i]);
		}
	}

	void resolve_dilations(void)
	{
		if( dilations.size() == 0 )
			for( unsigned i=0; i< get_numDataDim(); i++ )
				dilations.push_back(1);
	}
	void resolve_pads(void)
	{
		unsigned num_data_dim = get_numDataDim();
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
		unsigned num_data_dim = get_numDataDim();
		rv.push_back(get_X()->data_dim[0]);//batch size
		rv.push_back(get_W()->data_dim[0]);//"number of feature maps"

		for( unsigned dim=0, xdim=2;
		     dim < num_data_dim;
		     dim++, xdim++) {
			int outdim;
			// Not sure if the naming is correct. Here
			// kernel: the (number of) weights of the filter
			// filter: the spatial placement of the kernel weights
			// NB: 'dilation==1' is what is used for "no spacing in the filter"
			int filter_size=kernel_shape[dim];
			filter_size += (kernel_shape[dim]-1)*(dilations[dim]-1);

			// From ONNX Operators.md:
			// SAME_UPPER or SAME_LOWER mean pad the input so that the output spatial size match the input.
			// "match" here means "is equal".
			if( auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" )
				outdim = get_X()->data_dim[xdim];
			else if( auto_pad == "NOTSET" || auto_pad == "VALID") {
				//padded input
				int input_size = get_X()->data_dim[xdim] + pads[dim]+pads[dim+num_data_dim];
				// [ 0 1 2 3 4 5 6 7 8 9  ]
				//                |kern=3|
				// last output=7
				int last_out = input_size - filter_size;
				outdim = last_out / strides[dim] + 1;
			} else {
				ERROR("Invalid option for auto_pad attribute");
			}

			rv.push_back(outdim);
		}

		return rv;
	}

	// Does output channels map one-to-one to input channels.
	// This is only true for pooling filters.
	virtual bool direct_channel_map(void) const
	{
		return false;
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
		unsigned n_data_dims = get_numDataDim();
		unsigned batch_size = get_X()->data_dim[0];
		unsigned channels = get_X()->data_dim[1];
		unsigned maps=get_Y()->data_dim[1];

		/* Create various indexing strings. This makes generating the loops much cleaner,
		 * and makes possible the code sharing in child classes. */
		std::string x_idx = "[b][c]";
		std::string in_kern_idxs = "[b][c]";
		std::string y_idx = "[b][m]";
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			x_idx += "[i" + i_str + "]";
			y_idx += "[o" + i_str + "]";
			in_kern_idxs += "[ii" + i_str + "]";
		}

		/* Create the loops over batches and channels.
		 * In case this SpatialFilter has a weights input (w), this first loop is over
		 * output channels (M). Othervise input channels==outputchannels, and it is named C
		 */
		INDT_1 << "for( size_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		if( options.quantize ) {
			INDT_2 << "int32_t batch_min = INT32_MAX;" << std::endl;
			INDT_2 << "int32_t batch_max = INT32_MIN;" << std::endl;
		}
		if( direct_channel_map() )
			INDT_1 << "for( size_t m=0, c=0; m<" << maps << "; m++, c=m) {" << std::endl;
		else if( get_W() && group > 1 ) {
			INDT_1 << "size_t go = " << maps/group     << "; // output group size, i.e. maps/group" << std::endl;
			INDT_1 << "size_t gi = " << channels/group << "; // inptput group size, i.e. channels/group" << std::endl;
			INDT_1 << "for( size_t g=0; g<" << group << "; g++) {" << std::endl;
			INDT_1 << "for( size_t m=go*g; m<go*(g+1); m++) {" << std::endl;
		}
		else
			INDT_1 << "for( size_t m=0; m<" << maps << "; m++) {" << std::endl;


		// loop over outputs and inputs
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string o_idx = "o" + std::to_string(i);
			std::string i_idx = "i" + std::to_string(i);
			INDT_2 << "for( int32_t " << o_idx << "=0, ";
			   dst <<       i_idx << "=" << -pads[i] << "; ";
			   dst <<       o_idx << "<" << get_Y()->data_dim[2+i] << "; ";
			   dst <<       o_idx <<"++, "<< i_idx << "+=" << strides[i] << ") {" << std::endl;
		}

		print_output_cell_init(dst, y_idx);

		if (direct_channel_map())
			;
		else if( get_W() && group > 1 )
			INDT_3 <<   "for( size_t c=gi*g; c<gi*(g+1); c++ ) {" << std::endl;
		else    // same as above, just cleaner to read :)
			INDT_3 <<   "for( size_t c=0; c<" << channels << "; c++ ) {" << std::endl;

		std::string w_idx = "[m][c]";
		if (group != 1)
			w_idx = "[m][c-(gi*g)]";
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string idx = "k" + std::to_string(i);
			INDT_3 << "for( size_t " << idx << "=0; ";
			   dst <<       idx << "<" << kernel_shape[i] << "; ";
			   dst <<       idx <<"++ ) {" << std::endl;
			w_idx += "[" + idx + "]";
		}

		// check for out-of-input reading (i.e. read a pad)
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			INDT_4 <<  "int ii" << i_str << " = i" << i_str << "+k" << i_str <<" * " << dilations[i] <<";" << std::endl;
			INDT_4 <<  "if( ii" << i_str << "<0) continue;" << std::endl;
			INDT_4 <<  "if( ii" << i_str << ">=" << get_X()->data_dim[2+i] << ") continue;" << std::endl;
		}

		print_output_cell_calc(dst, in_kern_idxs, w_idx, y_idx);

		// close kernel loop
		for( unsigned i = 0; i<n_data_dims; i++)
			INDT_3 << "} /* k */" << std::endl;

		// close input channels loop when it is separate from output channels
		if( direct_channel_map() == false )
			INDT_3 << "} /* c */" << std::endl;
		print_output_cell_finalize(dst, y_idx);

		// close output loop
		for( unsigned i = 0; i<n_data_dims; i++)
			INDT_2 << "} /* o */" << std::endl;

		// close loops over batches and output channels
		INDT_1 << "} /* m */" << std::endl;
		if( direct_channel_map() == false && group > 1 )
			INDT_2 << "} /* g */" << std::endl;
		INDT_1 << "} /* b */" << std::endl;
	}
};
}

