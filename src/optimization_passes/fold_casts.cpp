/* This file is part of onnx2c.
 *
 * Implemented here is the 'fold_casts' optimization pass
 * that tires to remove Cast nodes.
 */
#include "graph.h"
#include <cassert>
#include <cstdint>

using namespace toC;

void Graph::fold_casts(void)
{
	LOG(DEBUG) << "Optimisation pass: fold casts"<< std::endl;
	std::vector<Node*> removed_nodes;

	// Loop over all Cast nodes
	for( auto n : nodes ) {
		if( n->op_name != "Cast" ) {
			LOG(TRACE) << n->onnx_name << " is not a Cast node, ignoring."<< std::endl;
			continue;
		}
		LOG(DEBUG) << "considering 'Cast' Node: " << n->onnx_name << std::endl;


		// If the Cast node's input has other users
		// the transformation becomes too difficult.
		// The input generating Predecessor node
		// would now need to generate two different
		// outputs, one for the folded cast, one of
		// the other user(s).
		// Skip folding these Cast nodes.
		assert(n->get_number_of_inputs() == 1);
		Tensor *input_tensor = n->get_input_tensor(0);
		Tensor *output_tensor = n->get_output_tensor(0);
		if( input_tensor->consumers.size() != 1 ) {
			LOG(DEBUG) << "  skipping. Input tensor has other users."<< std::endl;
			continue;
		}


		// Degenerate case where the graph input is directly the output.
		// This happens in unit tests at least, but other than that, sounds like an error.
		if( output_tensor->isIO && input_tensor->isIO ) {
			LOG(WARNING) << "   Cast output is graph output??" << std::endl;
			continue;
		}


		LOG(DEBUG) << "  folding away this Cast node."<< std::endl;
		// Modify the Predecessor node's output to
		// match the type of the Cast node's output.
		onnx::TensorProto_DataType cast_to_type;
		cast_to_type = output_tensor->data_type;
		input_tensor->data_type = cast_to_type;

		// Replace the Cast output tensor's users input
		// with the Predecessor node output. I.e. bypass
		// the cast node.
		for( auto cn : output_tensor->consumers ) {
			bool replaced;
			replaced = cn->replace_input(output_tensor, input_tensor );
			if( !replaced ) {
				LOG(FATAL) << output_tensor->name << " was not replaced" << std::endl;
			}
			else {
				std::erase(tensors, output_tensor);
				delete output_tensor;
			}
		}

		// Mark the now orphaned Cast node for removal
		removed_nodes.push_back(n);
	}

	for( auto rn : removed_nodes ) {
		std::erase(nodes, rn);
		delete rn;
	}
	LOG(TRACE) << "folding Cast nodes finished" << std::endl;
}


