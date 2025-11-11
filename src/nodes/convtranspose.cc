/* This file is part of onnx2c.
 *
 * ConvTranspose node.
 * Functionality acccording to ONNX documentation:
 * "The convolution transpose operator consumes an input tensor and a filter, and computes the output."
 *
 * Pytorch opens ConvTranspose2d up a bit more:
 * "This module can be seen as the gradient of Conv2d with respect to its input. It is also known as a
 *  fractionally-strided convolution or a deconvolution (although it is not an actual deconvolution
 *  operation as it does not compute a true inverse of convolution)".
 * With added links:
 *  - https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
 *  - https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
 *
 *
 * In the end, what is implemented in this file is the algorithm described in:
 * http://d2l.ai/chapter_computer-vision/transposed-conv.html
 *
  def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
 *
 * Since ONNX backend tests pass, this should be correct :)
 */
#include "convtranspose.h"

namespace toC {


void ConvTranspose::parseAttributes( onnx::NodeProto &node ) {
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
		else if( a.name() == "output_padding" )
			output_padding = parse_attribute_ints(a);
		else if( a.name() == "output_shape" )
			output_shape = parse_attribute_ints(a);
	}
}

void ConvTranspose::resolve_strides(void)
{
	unsigned num_data_dim = x->rank()-2;
	if( strides.size() == 0 )
		for( unsigned i=0; i<num_data_dim; i++)
			strides.push_back(1);
}

void ConvTranspose::resolve_kernel_shape(void)
{
	if( kernel_shape.size() == 0 ) {
		for( unsigned i=2; i<w->rank(); i++)
			kernel_shape.push_back(w->data_dim[i]);
	}
}

void ConvTranspose::resolve_dilations(void)
{
	unsigned num_data_dim = x->rank()-2;
	if( dilations.size() == 0 )
		for( unsigned i=0; i< num_data_dim; i++ )
			dilations.push_back(1);
}

void ConvTranspose::resolve_output_pads(void)
{
	unsigned num_data_dim = x->rank()-2;
	if( output_padding.size() == 0 )
		for( unsigned i=0; i<num_data_dim; i++)
			output_padding.push_back(0);
}


void ConvTranspose::resolve_output_shape(void)
{
	/* The documentation says:
	 * If the pads parameter is provided the shape of the output is calculated via the following equation:
	 * output_shape[i] = stride[i] * (input_size[i] - 1) + output_padding[i] +
	 *                  ((kernel_shape[i] - 1) * dilations[i] + 1) - pads[start_i] - pads[end_i]
	 */
	int num_data_dim = x->rank()-2;

	// Pads most likely must have been given if output_shape isn't. But its not required...
	//resolve_pads();
	if( pads.size() == 0 )
		pads.resize(num_data_dim*2, 0);

	for(unsigned d=2; d<x->rank(); d++)
	{
		unsigned i = d-2;
		unsigned os = strides[i] * (x->data_dim[d] -1);
		os += output_padding[i];
		os += (kernel_shape[i]-1) * dilations[i] + 1;
		os -= pads[i];
		os -= pads[i+num_data_dim];

		unsigned extra_pad = (x->data_dim[d] * strides[i]) - os;
		if( auto_pad == "SAME_UPPER" ) {
			pads[i+num_data_dim]++;
			os += extra_pad;
		}
		else if( auto_pad == "SAME_LOWER") {
			pads[i]++;
			os += extra_pad;
		}
		else
			; // Now what?

		output_shape.push_back(os);
	}
}


void ConvTranspose::resolve_convtranspose_pads(void)
{
	/* The documentation says:
	 * output_shape can also be explicitly specified in which case pads values are auto generated using these equations:
	total_padding[i] = stride[i] * (input_size[i] - 1) + output_padding[i] + ((kernel_shape[i] - 1) * dilations[i] + 1) - output_shape[i]
	If (auto_pads == SAME_UPPER): pads[start_i] = total_padding[i]/2; pads[end_i] = total_padding[i] - (total_padding[i]/2)
	Else: pads[start_i] = total_padding[i] - (total_padding[i]/2); pads[end_i] = (total_padding[i]/2).
	*/

	int num_data_dim = x->rank()-2;
	pads.resize(num_data_dim*2);
	// loop over the width, height or 3D equivalent dimensions.
	for(unsigned d=2; d<x->rank(); d++)
	{
		unsigned i = d-2;
		int total_padding = strides[i] * (x->data_dim[d] - 1);
		total_padding += output_padding[i];
		total_padding += (kernel_shape[i]-1)*dilations[i] + 1;
		total_padding -= output_shape[i];

		// this is needed either because:
		//  - there is a bug somewhere in onnx2c (ONNX backend tests pass with but not without this, though :))
		//  - it is implicitly assumed in ONNX documentation
		if( total_padding < 0 ) total_padding = 0;

		pads[i] = total_padding/2;
		pads[i+num_data_dim] = total_padding/2;

		if( total_padding % 2 == 0 )
			continue;
		if( auto_pad == "SAME_UPPER" )
			pads[i+num_data_dim]++;
		else
			pads[i]++;
	}
}

std::vector<int> ConvTranspose::calculate_output_size(void)
{
	std::vector<int> shape;
	shape.push_back(x->data_dim[0]); // batch
	shape.push_back(w->data_dim[1] * group); // maps
	for( auto s : output_shape )
		shape.push_back(s);
	return shape;
}

void ConvTranspose::resolve(void)
{
	x = get_input_tensor(0); // data
	name_input(0,"x");
	w = get_input_tensor(1); // weights
	name_input(1,"w");
	if( get_number_of_inputs() == 3 ) {
		b = get_input_tensor(2);
		name_input(2,"bias"); // 'b' is the batch looping index
	}
	else
		b = NULL;

	// ONNX backend test didn't have a test with groups :|
	if( group != 1 )
		ERROR("Unimplemented: groups in ConvTranspose");

	// Set defaults and calculate attributes
	resolve_strides();
	resolve_kernel_shape();
	resolve_dilations();
	resolve_output_pads();
	if( output_shape.size() == 0 ) {
		output_shape_given = false;
		resolve_output_shape();
	}
	else {
		output_shape_given = true;
		resolve_convtranspose_pads();
	}

	Tensor *rv = new Tensor;
	rv->data_dim = calculate_output_size();
	rv->data_type = x->data_type;
	register_output(rv, "y");
	y=rv;

}

// Print out source for node function body
void ConvTranspose::print(std::ostream &dst) const
{
	print_header_info_comment(dst);

	print_calculation(dst);
}

void ConvTranspose::print_header_info_comment(std::ostream &dst) const
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
	INDT_1 << " * output_padding: ";
		for( int o: output_padding)
			dst << o << " ";
		dst << std::endl;
	INDT_1 << " * output_shape: ";
		for( int o: output_shape)
			dst << o << " ";
		dst << std::endl;
	INDT_1 << " * output_shape explicitly given in ONNX model: " << (output_shape_given?"true":"false") << std::endl;
	INDT_1 << " */" << std::endl;
}

/*
 * NB: the channel/map/group code is copied from SpatialFilter.
 * The ONNX backend test don't test it.
 */
void ConvTranspose::print_calculation( std::ostream &dst) const
{
	unsigned n_data_dims = x->data_dim.size() -2;
	unsigned batch_size = x->data_dim[0];
	unsigned channels = x->data_dim[1];
	unsigned maps=y->data_dim[1];

	// Create various indexing strings. This makes generating the loops much cleaner.
	std::string x_idx = "[b][c]";
	std::string w_idx = "[c][m]"; // TODO: in case of groups, change c to something else
	std::string y_idx = "[b][m]";
	for( unsigned i = 0; i<n_data_dims; i++) {
	std::string i_str = std::to_string(i);
		x_idx += "[i" + i_str + "]";
		y_idx += "[o" + i_str + "]";
		w_idx += "[k" + i_str + "]";
	}

	// clear output - required by the algorithm
	INDT_1 << "memset(y, 0," << y->data_num_elem()*y->data_elem_size() << ");" << std::endl << std::endl;

	// Create the loops over batches and maps (output channels).
	INDT_1 << "for( size_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
	INDT_1 << "for( size_t m=0; m<" << maps << "; m++) {" << std::endl;

	// loop inputs
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string i_idx = "i" + std::to_string(i);
		INDT_2 << "for( size_t " << i_idx << "=0; ";
		   dst <<       i_idx << "<" << x->data_dim[2+i] << "; ";
		   dst <<       i_idx <<"++) {" << std::endl;
	}

	// Loop over input channels
	INDT_3 <<   "for( size_t c=0; c<" << channels << "; c++ ) {" << std::endl;


	// Generate loops over outputs and kernel indices. Something like:
	// "for ( k0=0, o0=0; k0<3; k0++, o0+=1){"
	// This calculates the N-dimensional dot product of input and kernel.
	//
	// The oN variable can be negative here when using padding. The naming is a bit
	// confusing. o (output) is an index to Y (the tensor that this node writes to)
	// but the padding named 'pad' is applied to the output (and not to the
	// input as in the "forwared" spatial filters). On top of this, there is the
	// concept and attribute of 'output_pad', which seems to be used only to calculate
	// the size of the output. I.e. 'pad's are excluded from the output, 'output_pad's
	// are included.
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string k_idx = "k" + std::to_string(i);
		std::string o_idx = "o" + std::to_string(i);
		std::string i_idx = "i" + std::to_string(i);
		std::string o_start = i_idx + "*" + std::to_string(strides[i]) + "-" + std::to_string(pads[i]);
		std::string o_incr = o_idx + "+=" + std::to_string(dilations[i]);

		INDT_3 << "for( int32_t " << k_idx << "=0, " << o_idx << "=" << o_start << "; ";
			   dst <<       k_idx << "<" << kernel_shape[i] << "; ";
			   dst <<       k_idx <<"++, " << o_incr << ") {" << std::endl;
	}

	// Skip write if the index into Y is a padding
	for( unsigned i = 0; i<n_data_dims; i++) {
		std::string i_str = std::to_string(i);
		INDT_4 <<  "if( o" << i_str << "<0) continue;" << std::endl;
		INDT_4 <<  "if( o" << i_str << ">=" << output_shape[i] << ") continue;" << std::endl;
	}

	// Do the actual calculation
	INDT_4 << "y" << y_idx;
	dst <<    " += x" << x_idx;
	dst <<    " * w" << w_idx;
	dst << ";" << std::endl;

	// close kernel loop
	for( unsigned i = 0; i<n_data_dims; i++)
		INDT_3 << "} /* k */" << std::endl;

	// close input channels loop when it is separate from output channels
	INDT_3 << "} /* c */" << std::endl;

	// close output loop
	for( unsigned i = 0; i<n_data_dims; i++)
		INDT_2 << "} /* o */" << std::endl;

	// close loops over batches and output channels
	INDT_1 << "} /* m */" << std::endl;
	INDT_1 << "} /* b */" << std::endl;
    
    // YK: bias should be added only once
    if( b ) {
        // Create the loops over batches and maps (output channels).
        INDT_1 << "for( size_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
        INDT_2 << "for( size_t m=0; m<" << maps << "; m++) {" << std::endl;
        for( unsigned i = 0; i<n_data_dims; i++) {
            std::string i_str = std::to_string(i);
            INDT_3 << "for( size_t o"<<i_str<<"=0; o"<<i_str<<"<" << output_shape[i] << "; o"<<i_str<<"++) {" << std::endl;
            
        }
        INDT_4 << "y" << y_idx;
        dst << " += bias[m]";
        dst << ";" << std::endl;
        
        for( unsigned i = 0; i<n_data_dims; i++) {
            std::string i_str = std::to_string(i);
            INDT_3 << "} /* o"<<i_str<<" */" << std::endl;
        }
        
        
        INDT_2 << "} /* m */" << std::endl;
        INDT_1 << "} /* b */" << std::endl;
    }
        
}

} // namespace

