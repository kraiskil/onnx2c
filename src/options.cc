/* This file is part of onnx2c.
 */

#include "options.h"
#include "args.hxx"
#include "error.h"
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

void initialize_logging(void)
{
	AixLog::Severity s;
	switch(options.logging_level)
	{
	case 4: s = AixLog::Severity::trace;   break;
	case 3: s = AixLog::Severity::debug;   break;
	default:
	case 2: s = AixLog::Severity::info;    break;
	case 1: s = AixLog::Severity::warning; break;
	case 0: s = AixLog::Severity::error;   break;
	}
	AixLog::Log::init<AixLog::SinkCerr>(s);
}

void store_define_option(const std::string &opt)
{
	auto delim_pos = opt.find(':', 0 );
	if( delim_pos == std::string::npos )
		ERROR("bad command line argument for the '-d' option");

	std::string name = opt.substr(0, delim_pos);
	if( name.size() < 1 )
		ERROR("bad command line argument for the '-d' option");

	std::string val = opt.substr(delim_pos+1, std::string::npos);
	if( val.size() < 1 )
		ERROR("bad command line argument for the '-d' option");

	uint32_t val_num;
	try {
		val_num = std::stoul(val);
	}
	catch( std::exception& e ) {
		ERROR("bad command line argument for the '-d' option");
	}

	options.dim_defines[name] = val_num;
}

void parse_cmdline_options(int argc, const char *argv[])
{
	args::ArgumentParser parser("Generate C code from an ONNX graph file.");
	args::Flag avr(parser, "avr", "Target AVR-GCC", {'a', "avr"});
	args::ValueFlagList<std::string> define(parser, "dim:size", "Define graph input dimension. Can be given multiple times", {'d', "define"});
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
	if (loglevel) {options.logging_level = args::get(loglevel); }

	// initialize logging as soon as possible, so logging is available in parsing the options too
	initialize_logging();

	if (quantize) { options.quantize = true; }
	if (avr) { options.target_avr = true; }
	if (define) {
		for (const auto &d: args::get(define)) {
			store_define_option(d);
		}
	}
	if (input) { options.input_file = args::get(input); }
	if (options.input_file == "" ) { std::cerr << "No input file given"; hint_at_help_and_exit(); }
}

