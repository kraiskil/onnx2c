/* This file is part of onnx2c.
 */
#include <iostream>
#include <fstream>

#include "onnx.pb.h"

#include "graph.h"
#include "options.h"
#include "tensor.h"

int main(int argc, const char *argv[])
{
	onnx::ModelProto onnx_model;

	parse_cmdline_options(argc, argv);

	std::ifstream input(options.input_file);
	if (!input.good()) {
		std::cerr << "Error opening input file: \"" << options.input_file << "\""  << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}
	if (input.peek() == EOF) {
		ERROR("\"" << options.input_file << "\" is empty");
	}
	if (!onnx_model.ParseFromIstream(&input)) {
		ERROR("\"" << options.input_file << "\" is not a valid ONNX model");
	}

	std::cout.precision(20);
	toC::Graph toCgraph(onnx_model);
	if( options.opt_fold_casts )
		toCgraph.fold_casts();
	if( options.opt_unionize )
		toCgraph.unionize_tensors();
	toCgraph.set_no_globals(options.no_globals);

	if (options.only_init) {
		toCgraph.print_initialization(std::cout);
	} else {
		toCgraph.print_source(std::cout);
	}
}

