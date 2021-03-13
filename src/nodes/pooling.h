/* This file is part of onnx2c.
 *
 * Pooling layer common.
 * Max and AveragePool have some complicated
 * output size calculations.
 */
#pragma once
#include "spatialfilter.h"

namespace toC {

class Pooling : public SpatialFilter {

	public:

	Pooling() : SpatialFilter() {
		ceil_mode=0;
		count_include_pad=0;
		storage_order=0;
	}

	/* Attributes */
	int ceil_mode;
	int count_include_pad;
	int storage_order;
	
	virtual void parseAttributes( onnx::NodeProto &node ) override {

		SpatialFilter::parseAttributes(node);

		for( const auto& a : node.attribute() ) {
			if( a.name() == "ceil_mode" )
				ceil_mode = parse_attribute_int(a);
			else if( a.name() == "count_include_pad" )
				count_include_pad = parse_attribute_int(a);
			else if( a.name() == "storage_order" )
				storage_order = parse_attribute_int(a);
		}
	}
	virtual std::vector<int> resolve_output_size(void)
	{
		std::vector<int> rv;
		rv.push_back(x->data_dim[0]);//batch
		rv.push_back(x->data_dim[1]);//channel
	
		unsigned data_dims = x->data_dim.size()-2;
		std::vector<int> pad_shapes;
		for( unsigned i=0; i<data_dims; i++ ) {
			pad_shapes.push_back(pads[i]+pads[data_dims+i]);
		}
		// Calculate output shape. Pads are now calculated
		// for those auto_pad modes that need them.
		for( unsigned i=2; i<x->data_dim.size(); i++ ) {
			int d;
			int in_dim = x->data_dim[i];
			int kernel = kernel_shape[i-2];
			int dilation = dilations.size()==0 ? 1 : dilations[i-2];
			int stride = strides[i-2];
			if ( auto_pad == "NOTSET" ) {
				//int pad_sh = pad_shapes[i-2];
				int pad_sh = pad_shapes[i-2];
				if (ceil_mode)
					d = ceil((float)(in_dim + pad_sh - ((kernel - 1) * dilation + 1)) / stride + 1);
				else
					d = floor((float)(in_dim + pad_sh - ((kernel  - 1) * dilation + 1)) / stride + 1);
			}

			else if( auto_pad == "VALID" )
				d = ceil((float)( in_dim - ((kernel - 1) *dilation + 1) +1) /  stride );

			else // auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER"
				d = ceil( (float)in_dim / stride );

			rv.push_back(d);
		}
		return rv;
	}

	// The auto_pad mess: 
	// pads are needed to calculate output shape, but output shape is needed to calculate pads
	// Run this after resolve_output_size() to patch up
	void update_pads(void)
	{
		if ( auto_pad == "NOTSET" )
			return;
		if ( auto_pad == "VALID" )
			return;

		unsigned data_dims = x->data_dim.size()-2;

		// Calculate pads for the "SAME_*" cases that need the output shape 
		for(unsigned i=0; i<data_dims; i++) {
			// NB: onnx2c architecture does not allow figuring out the output shape at this stage
			// (especially since the onnx spec says it is a function of input, strides, pads &c).
			// The auto_pad attribute for AveragePool is deprecated anyway. Probably just for this confusion.
			// This tries to be some sort of band-aid: assume the output size is the same as input size
			// which is the usual(?) reason to use "same" padding on the network design level. 
			int input_size = x->data_dim[i+2];
			int output_size = y->data_dim[i+2];
			int pad_shape = (output_size - 1) * strides[i] + (( kernel_shape[i] -1) * dilations[i]+1) - input_size; 
			pads[i] = pad_shape/2;
			pads[i+data_dims] = pad_shape/2;
			if( pad_shape & 1 ) {
				if( auto_pad == "SAME_LOWER" )
					pads[i]++;
				else
					pads[i+data_dims]++;
			}
		}
	}

};
}

