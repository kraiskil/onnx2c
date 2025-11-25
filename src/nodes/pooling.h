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
		count_include_pad=0;
		storage_order=0;
	}

	/* Attributes */
	int count_include_pad;
	int storage_order;

	bool direct_channel_map(void) const override
	{
		return true;
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		SpatialFilter::parseAttributes(node);

		for( const auto& a : node.attribute() ) {
			if( a.name() == "count_include_pad" )
				count_include_pad = parse_attribute_int(a);
			else if( a.name() == "storage_order" )
				storage_order = parse_attribute_int(a);
		}
	}
};
}

