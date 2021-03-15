
#include <iostream>
#include <fstream>

#include "onnx.pb.h"

#include "graph.h"
#include "tensor.h"

bool quantize=false;

int main(int argc, char *argv[])
{
	onnx::ModelProto onnx_model;

	if (argc < 2) {
		std::cerr << "Usage: onnx2c [-v] [-quantize] <.onnx-file>" << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	int fileargno = 1;
	bool verbose_mode=false;


	if ( strncmp(argv[fileargno], "-v", 2 ) == 0 ) {
		verbose_mode = true;
		fileargno++;
	}

	if ( strncmp(argv[fileargno], "-quantize", 9 ) == 0 ) {
		quantize = true;
		fileargno++;
	}


	std::ifstream input(argv[fileargno]);
	if (!input.good()) {
		std::cerr << "Error opening input file" << std::endl;
		exit(1); //TODO: check out error numbers for a more accurate one
	}

	onnx_model.ParseFromIstream(&input);

	std::cout.precision(20);
	toC::Graph toCgraph(onnx_model, verbose_mode);
	toCgraph.print_source(std::cout);
}

