/* This file is part of onnx2c.
 *
 * MaxPool
 * Picks maximum value from the elements that
 * are under the kernel.
 *
 * Current implementation is likely non-optimal:
 * padding checks are done in the innermost
 * loop. The C compiler need to do real magic
 * to optimize that...
 */
#include <cmath>

namespace toC {

class MaxPool : public Node {
	public:
	MaxPool() {
		op_name = "MaxPool";
		ceil_mode = 0;
		storage_order=0;
		auto_pad = "NOTSET";
	}
	/* MaxPool node specific attributes */
	int ceil_mode;
	std::vector<int> dilations;
	std::vector<int> kernel_shape;
	std::vector<int> pads;
	int storage_order;
	std::vector<int> strides;
	std::string auto_pad;

	std::vector<int> pad_shapes; // pad_shapes[i] = "sum of pads along axis i"

	void parseAttributes_auto_pad( const onnx::AttributeProto &a ) {
		auto_pad = a.s();
	}

	void parseAttributes_ceil_mode( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INT )
			ERROR("Wrong attribute type for MaxPool attribute 'ceil_mode'");
		ceil_mode = a.i();
	}

	void parseAttributes_dilations( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for MaxPool attribute 'dilations'");

		for( auto i : a.ints() ) {
			dilations.push_back(i);
		}
	}


	void parseAttributes_kernel_shape( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for MaxPool attribute 'kernel_shape'");

		for( auto i : a.ints() ) {
			kernel_shape.push_back(i);
		}
	}

	void parseAttributes_pads( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for MaxPool attribute 'pads'");

		for( auto i : a.ints() ) {
			pads.push_back(i);
		}
	}

	void parseAttributes_storage_order( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INT )
			ERROR("Wrong attribute type for MaxPool attribute 'storage_order'");
		storage_order = a.i();
	}

	void parseAttributes_strides( const onnx::AttributeProto &a ) {
		if( a.type() != onnx::AttributeProto_AttributeType_INTS )
			ERROR("Wrong attribute type for MaxPool attribute 'kernel_strides'");

		for( auto i : a.ints() ) {
			strides.push_back(i);
		}
	}

	virtual void parseAttributes( onnx::NodeProto &node ) override {

		for( const auto a : node.attribute() ) {
			if( a.name() == "auto_pad" )
				parseAttributes_auto_pad(a);
			else if( a.name() == "ceil_mode" )
				parseAttributes_ceil_mode(a);
			else if( a.name() == "dilations" )
				parseAttributes_dilations(a);
			else if( a.name() == "kernel_shape" )
				parseAttributes_kernel_shape(a);
			else if( a.name() == "pads" )
				parseAttributes_pads(a);
			else if( a.name() == "storage_order" )
				parseAttributes_storage_order(a);
			else if( a.name() == "strides" )
				parseAttributes_strides(a);
		}
	}

	virtual void print(std::ostream &dst) const
	{
		if( inputs.size() != 1 )
			ERROR("wrong number of inputs to MaxPool");
		if( outputs.size() != 1 )
			ERROR("wrong number of outputs from MaxPool");

		dst << "\t/* MaxPool" << std::endl;
		dst << "\t *" << std::endl;
		dst << "\t * auto_pad: " << auto_pad << std::endl;
		dst << "\t * ceil_mode: " << ceil_mode <<std::endl;
		dst << "\t * dilations: ";
			for( int d: dilations )
				dst << d << " ";
		dst << std::endl;
		dst << "\t * kernel_shape: ";
			for( int k: kernel_shape )
				dst << k << " ";
		dst << std::endl;
		dst << "\t * pads: ";
			for( int p: pads)
				dst << p << " ";
		dst << std::endl;
		dst << "\t * storage_order: " << storage_order <<std::endl;
		dst << "\t * strides: ";
			for( int s: strides)
				dst << s << " ";
		dst << std::endl <<  "\t */" << std::endl;

		int batch_size = inputs[0]->data_dim[0];
		int channels = inputs[0]->data_dim[1];
		unsigned n_data_dims = inputs[0]->data_dim.size()-2;
		std::string type = inputs[0]->data_type_str();
		std::string type_min_value;
		std::string in = inputs[0]->cname();
		std::string out = outputs[0]->cname();

		if( type == "float" )
			type_min_value = "-FLT_MAX";
		else if( type == "uint8_t" )
			type_min_value = "0";
		else
			ERROR("Unimplemented: minimum value for this type");

		std::string in_idxs = "[b][c]";
		std::string in_kern_idxs = "[b][c]";
		std::string out_idxs = "[b][c]";
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			in_idxs += "[i" + i_str + "]";
			out_idxs += "[o" + i_str + "]";
			in_kern_idxs += "[ii" + i_str + "]";
		}

		// loop over batches and channels
		dst<<"\t"        << "for( int32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
		dst<<"\t"      <<   "for( int32_t c=0; c<" << channels << "; c++ ) {" << std::endl;

		// loop over outputs and inputs
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string o_idx = "o" + std::to_string(i);
			std::string i_idx = "i" + std::to_string(i);
			dst << "\t\t" << "for( int32_t " << o_idx << "=0, ";
			dst <<               i_idx << "=" << -pads[i] << "; ";
			dst <<               o_idx << "<" << outputs[0]->data_dim[2+i] << "; ";
			dst <<               o_idx <<"++, "<< i_idx << "+=" << strides[i] << ") {" << std::endl;
		}

		dst<<"\t\t\t"  <<       type << " curmax = " << type_min_value << ";" << std::endl;

		// loop over kernel
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string idx = "k" + std::to_string(i);
			dst << "\t\t\t" << "for( uint32_t " << idx << "=0; ";
			dst <<               idx << "<" << kernel_shape[i] << "; ";
			dst <<               idx <<"++ ) {" << std::endl;
		}

		// check for out-of-input reading (i.e. read a pad)
		for( unsigned i = 0; i<n_data_dims; i++) {
			std::string i_str = std::to_string(i);
			dst<<"\t\t\t\t"  <<  "int ii" << i_str << " = i" << i_str << "+k" << i_str <<" + " << dilations[i] <<"-1;"<<std::endl;
			dst<<"\t\t\t\t"  <<  "if( ii" << i_str << "<0) continue;" << std::endl;
			dst<<"\t\t\t\t"  <<  "if( ii" << i_str << ">=" << inputs[0]->data_dim[2+i] << ") continue;" << std::endl;
		}


		dst<<"\t\t\t\t"<<         "curmax = MAX( curmax, " << in << in_kern_idxs << ");" <<std::endl;

		// close kernel loop
		for( unsigned i = 0; i<n_data_dims; i++)
			dst<<"\t\t\t}" << std::endl;

		dst<<"\t\t\t"  <<       out << out_idxs << "= curmax;" << std::endl;

		// close output loop
		for( unsigned i = 0; i<n_data_dims; i++)
			dst<<"\t\t}" << std::endl;

		// close loops over batches and cahannels
		dst<<"\t}" << std::endl;
		dst<<"\t}" << std::endl;
	}
 
	virtual void resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
	{
		const Tensor *x = inputs[0];

		if( !(  typeConstraint_plainFloatingPoints(x)
		      ||typeConstraint_8bit(x)) )
			ERROR("Incorrect input for node"); 

		if( x->data_dim[0] != 1 )
			ERROR("Unimplemented: MaxPool batches bigger than 1");


		if( kernel_shape.size() == 0 )
			ERROR("MaxPool: kernel_shape not provided");

		if( storage_order != 0 )
			ERROR("Unimplemented: column-major storage_order");

		if( strides.size() == 0 )
			for( unsigned i=0; i< x->data_dim.size(); i++ )
				strides.push_back(1);

		if( dilations.size() == 0 )
			for( unsigned i=0; i< x->data_dim.size(); i++ )
				dilations.push_back(1);

		unsigned data_dims = x->data_dim.size()-2;

		// if 'pads' attribute not given, fill with defaults
		// VALID or NOTSET. Former always means no padding, latter means explicit
		// or if not given, no padding
		// For "SAME_*" the output dimensions must be calculated first,
		// but the others need the pads to calculate the output dimensions...
		if( pads.size() == 0 ) {
			if( auto_pad == "NOTSET" || auto_pad == "VALID" ) {
				pads.resize(data_dims * 2);
				for( auto &p : pads )
					p=0;
			}
		}

		if( pads.size() != 0 ) {
			for( unsigned i=0; i<data_dims; i++ ) {
				pad_shapes.push_back(pads[i]+pads[data_dims+i]);
			}
		}

		Tensor *rv = new Tensor;
		rv->data_dim.push_back(x->data_dim[0]); //batch size
		rv->data_dim.push_back(x->data_dim[1]); //num channels

		// Calculate output shape. Pads are now calculated
		// for those auto_pad modes that need them.
		uint64_t rv_num_elem = x->data_dim[0] * x->data_dim[1];
		for( unsigned i=2; i<x->data_dim.size(); i++ ) {
			int d;
			int in_dim = x->data_dim[i];
			int kernel = kernel_shape[i-2];
			int dilation = dilations.size()==0 ? 1 : dilations[i-2];
			int stride = strides[i-2];
			if ( auto_pad == "NOTSET" ) {
				int pad_sh = pad_shapes[i-2];
				if (ceil_mode)
					d = ceil((float)(in_dim + pad_sh - ((kernel - 1) * dilation + 1)) / stride + 1);
				else
					d = floor((float)(in_dim + pad_sh - ((kernel  - 1) * dilation + 1)) / stride + 1);
			}

			// output_spatial_shape[i] = ceil((input_spatial_shape[i] -
			//              ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
			else if( auto_pad == "VALID" )
				d = ceil((float)( in_dim - ((kernel - 1) *dilation + 1) +1) /  stride );

			// output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
			else if( auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER" )
				d = ceil( (float)in_dim / stride );

			rv->data_dim.push_back(d);
			rv_num_elem *= d;
		};




		// Calculate pads for the "SAME_*" cases that need the output shape 
		if( pads.size() == 0 ) {
			pads.resize(data_dims * 2);
			for(unsigned i=0; i<data_dims; i++) {
				//pad_shape[i] = (output_spatial_shape[i] - 1) *
				//             strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)
				//              - input_spatial_shape[i]
				// NB: onnx2c architecture does not allow figuring out the output shape at this stage
				// (especially since the onnx spec says it is a function of input, strides, pads &c).
				// The auto_pad attribute for MaxPool is deprecated anyway. Probably just for this confusion.
				// This tries to be some sort of band-aid: assume the output size is the same as input size
				// which is the usual(?) reason to use "same" padding on the network design level. 
				int input_size = x->data_dim[i+2];
				int output_size = rv->data_dim[i+2];
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
			for( unsigned i=0; i<data_dims; i++ ) {
				pad_shapes.push_back(pads[i]+pads[data_dims+i]);
			}
		}
		if( pads.size() != 2*data_dims ) {
			ERROR("Pads size mismatch!");
		}

		rv->data_type = x->data_type;
		outputs.push_back(rv);
		//TODO: also push out the optional Indices tensor. But this needs some fixes in Graph. And a test case
		//      that uses multiple outputs.
	}
};
}
