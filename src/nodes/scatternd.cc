/* This file is part of onnx2c.
 *
 * ScatterND node.
 * This copies the input to output as is, updating parts of the
 * output as specified by the 'indices' input.
 */
#include "scatternd.h"

using namespace toC;

void ScatterND::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "reduction" )
			reduction = parse_attribute_string(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node ScatterND/" << onnx_name << std::endl;
	}
}


void ScatterND::resolve(void)
{
	if (inputs.size() != 3) {
		ERROR("Wrong number of inputs to ScatterND");
	}
	data = inputs[0];
	indices = inputs[1];
	updates = inputs[2];
	register_input(data, "data");
	register_input(indices, "indices");
	register_input(updates, "updates");

	Tensor *t = new Tensor;
	t->data_dim = data->data_dim;
	t->data_type = data->data_type;
	output = t;
	outputs.push_back(t);
	register_output(t, "output");
}


void ScatterND::print(std::ostream &dst) const
{

	unsigned k = indices->data_dim[indices->rank()-1];
	std::string data_op="=";
	if( reduction == "add" )
		data_op = "+=";
	else if( reduction == "mul" )
		data_op = "*=";


	INDT_1 << "/* ScatterND */" << std::endl;

	// Bulk copy the input to output. After this, the content of updates
	// is scattered on top of the output.
	INDT_1 << "memcpy(output, data, sizeof(" << output->data_type_str() << ")*" << output->data_num_elem() << ");" << std::endl;

	std::string u_idxs; // first part of the index string to 'updates' & the "meta" index string to 'indices'
	std::string o_idxs; // index string to to output

	// Create the first part of u_idxs
	unsigned i;
	for( i=0; i< indices->rank()-1 ; i++) {
		std::string idxstr = "idx"+std::to_string(i);
		INDT_1 << "for(uint64_t " <<idxstr<<" = 0; ";
		  dst  << idxstr << " < " << indices->data_dim[i] << "; ";
                  dst  << idxstr <<"++) {" << std::endl;

		u_idxs += "["+idxstr+"]";
	}

	// Create the indirect indexing (into 'output') variables
	for( i=0; i< k ; i++) {
		std::string istr=std::to_string(i);
		o_idxs += "[p" + istr + "]";   // start of index string to "output"
		INDT_2 << "unsigned p" << istr << " = indices" << u_idxs << "[" + istr + "];" << std::endl;
	}

	// Create the latter parts of the indexing strings + loops
	for( ; i<data->rank(); i++) {
		std::string i_str = std::to_string(i);
		std::string i_idx;
		std::string o_idx = "o" + i_str+"";

		u_idxs += "[" + o_idx + "]";
		o_idxs += "[" + o_idx + "]";

		INDT_2 << "for( uint32_t " << o_idx << "=0; ";
		   dst <<       o_idx << "<" << output->data_dim[i] << "; ";
		   dst <<       o_idx <<"++) {" << std::endl;
	}

	INDT_3 << "output"<< o_idxs << data_op << " updates" << u_idxs << ";" << std::endl;

	for( i = 0; i<indices->rank()-1; i++)
		INDT_2 << "}" << std::endl;

	for(i=k ; i < data->rank(); i++)
		INDT_1 << "}" << std::endl;
}

