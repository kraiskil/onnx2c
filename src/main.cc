
#include <iostream>
#include <fstream>

#include "onnx.pb.h"

#include "graph.h"
#include "tensor.h"


int main(int argc, char *argv[])
{
	onnx::ModelProto onnx_model;

	if (argc != 2) {
		std::cerr << "Usage: onnx2c <.onnx-file>" << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	std::ifstream input(argv[1]);
	if (!input.good()) {
		std::cerr << "Error opening input file" << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	onnx_model.ParseFromIstream(&input);

	toC::Graph toCgraph(onnx_model);
	toCgraph.print_source(std::cout);
}

