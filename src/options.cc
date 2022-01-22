/* This file is part of onnx2c.
 */

#include "options.h"
#include "args.hxx"

#include <iostream>

struct onnx2c_opts options;


void parse_cmdline_options(int argc, const char *argv[])
{
	args::ArgumentParser parser("Generate C code from an ONNX graph file.");
	args::Flag quantize(parser, "quantize", "Quantize network (EXPERIMENTAL!)", {'q', "quantize"});
	args::Flag avr(parser, "avr", "Target AVR-GCC", {'a', "avr"});
	args::ValueFlag<int> loglevel(parser, "level", "Logging verbosity. 0(none)-4(all)", {'l',"log"});
	args::Positional<std::string> input(parser, "input", "ONNX file to process");
	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help &)
	{
		std::cout << parser;
	}
	catch (args::ParseError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
	}
	catch (args::ValidationError &e)
	{
		std::cerr << e.what() << std::endl;
		std::cerr << parser;
	}
	if (quantize) { options.quantize = true; }
	if (avr) { options.target_avr = true; }
	if (input) { options.input_file = args::get(input); }
	if (loglevel) {options.logging_level = args::get(loglevel); }
	else options.logging_level = 2;
}
