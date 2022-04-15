/* This file is part of onnx2c.
 */

#include "options.h"
#include "args.hxx"
#include "timestamp.h"

#include <iostream>

struct onnx2c_opts options;

void hint_at_help_and_exit(void)
{
	std::cerr << ", use option flag '-h' to show help." << std::endl;
	exit(1);
}

void print_version_and_exit(void)
{
	std::cout << "Git branch: " << git_branch_str;
	std::cout << " - Commit: " << git_short_hash_str;
	std::cout << " - Build time: " << build_time_str << "(UTC)" << std::endl;
	exit(0);
}

void parse_cmdline_options(int argc, const char *argv[])
{
	args::ArgumentParser parser("Generate C code from an ONNX graph file.");
	args::Flag avr(parser, "avr", "Target AVR-GCC", {'a', "avr"});
	args::ValueFlag<int> loglevel(parser, "level", "Logging verbosity. 0(none)-4(all)", {'l',"log"});
	args::Flag help(parser, "help", "Print this help text.", {'h',"help"});
	args::Flag quantize(parser, "quantize", "Quantize network (EXPERIMENTAL!)", {'q', "quantize"});
	args::Flag version(parser, "version", "Print onnx2c version", {'v', "version"});
	args::Positional<std::string> input(parser, "input", "ONNX file to process");
	try
	{
		parser.ParseCLI(argc, argv);
	}
	catch (args::Help &)
	{
		// TODO: this catch case is from the ArgumentParser example. What do we catch here?
		hint_at_help_and_exit();
	}
	catch (args::ParseError &e)
	{
		std::cerr << e.what();
		hint_at_help_and_exit();
	}
	catch (args::ValidationError &e)
	{
		// TODO: this catch case is from the ArgumentParser example. What do we catch here?
		std::cerr << e.what();
		hint_at_help_and_exit();
	}

	if (help) { std::cout << parser; exit(0); }
	if (version) { print_version_and_exit(); }

	if (quantize) { options.quantize = true; }
	if (avr) { options.target_avr = true; }
	if (input) { options.input_file = args::get(input); }
	if (loglevel) {options.logging_level = args::get(loglevel); }
	else options.logging_level = 2;
	if (options.input_file == "" ) { std::cerr << "No input file given"; hint_at_help_and_exit(); }
}
