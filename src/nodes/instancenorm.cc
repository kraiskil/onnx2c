/* This file is part of onnx2c.
 *
 * InstanceNormalization node.
 * When implementing a new node, use this template
 * as a starting point.
 * Replace all occurances of InstanceNormalization in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */

#include "instancenorm.h"

namespace toC {

void InstanceNormalization::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "epsilon" )
			epsilon = parse_attribute_float(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node InstanceNormalization/" << onnx_name << std::endl;
	}
}


void InstanceNormalization::resolveOutput(const std::vector< const Tensor*> &inputs, std::vector<Tensor *> &outputs)
{
	input = inputs[0];
	register_input(input, "input");

	scale = inputs[1];
	register_input(scale, "scale");

	B = inputs[2];
	register_input(B, "B");

	Tensor *t = new Tensor;
	t->data_dim = input->data_dim;
	t->data_type = input->data_type;
	output = t;
	outputs.push_back(t);
	register_output(t, "output");
}


void InstanceNormalization::print(std::ostream &dst) const
{
	INDT_1 << "/* InstanceNormalization" << std::endl;
	INDT_1 << " */" << std::endl << std::endl;

	INDT_1 << "float epsilon = " << epsilon << ";" << std::endl;

	int batch_size =input->data_dim[0];
	int num_chan =input->data_dim[1];
	std::string type = input->data_type_str();


	INDT_1 << "for( int32_t b=0; b<" << batch_size << "; b++ ) {" << std::endl;
	INDT_1 << "for( int32_t c=0; c<" << num_chan << "; c++ ) {" << std::endl;


	std::string idxs = "[b][c]";
	unsigned instance_size=1;

	// First calculate mean & variance over instance
	// Other ways of calculating variance:
	// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
	dst << std::endl;
	INDT_2 << type << " mean =  0;" << std::endl;
	INDT_2 << type << " sqmean =  0;" << std::endl;
	for( unsigned i = 2; i<input->data_dim.size(); i++) {
		std::string idx = "i" + std::to_string(i);
		INDT_2 << "for( uint32_t " << idx << "=0; ";
		dst <<            idx << "<" << input->data_dim[i] << "; ";
		dst <<            idx <<"++ ) {" << std::endl;

		idxs += "[i" + std::to_string(i) + "]";
		instance_size *= input->data_dim[i];
	}
	 INDT_3 << type << " d = input" << idxs << ";" <<std::endl;
	 INDT_3 << "mean += d;" << std::endl;
	 INDT_3 << "sqmean += d*d;" << std::endl;

	for( unsigned i = 2; i<input->data_dim.size(); i++)
		INDT_2 << "}" << std::endl;

	INDT_2 << "mean /= " << instance_size << ";" << std::endl;
	INDT_2 << "sqmean /= " << instance_size << ";" << std::endl;
	INDT_2 << type << " var = sqmean - mean*mean;" << std::endl;
	dst << std::endl;


	// Now loop over the instance again, normalizing
	for( unsigned i = 2; i<input->data_dim.size(); i++) {
		std::string idx = "i" + std::to_string(i);
		INDT_2 << "for( uint32_t " << idx << "=0; ";
		dst <<            idx << "<" << input->data_dim[i] << "; ";
		dst <<            idx <<"++ ) {" << std::endl;
	}
	 INDT_3 << type << " d = input" << idxs << ";" <<std::endl;
	 INDT_3 << "output"<< idxs << " = scale[c] * (d-mean) / sqrt(var + epsilon) + B[c];" << std::endl;

	for( unsigned i = 2; i<input->data_dim.size(); i++)
		INDT_2 << "}" << std::endl;


	// close channel & batch
	INDT_1 << "}" << std::endl;
	INDT_1 << "}" << std::endl;
}
}

