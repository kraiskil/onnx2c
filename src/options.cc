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

void print_optimization_passes(void)
{
	std::cout << "Available optimization passes:" << std::endl;
	std::cout << " - 'unionize' (defaut:on)" << std::endl;
	std::cout << " - 'fold_casts' (defaut:on)" << std::endl;
	std::cout << " - 'none' (disable all optimization passes)" << std::endl;
}

void store_optimization_passes(const std::string &opt)
{
	LOG(TRACE) << "Parsing optimizations: " << opt << std::endl;

	// disable all optimizations (i.e. override the default settings)
	// then enable those that were requested
	options.opt_unionize=false;
	options.opt_fold_casts=false;
	if( opt == "none" )
	{
		LOG(TRACE) << "Disabling all optimizations: " << opt << std::endl;
		return;
	}

	if( opt == "help" )
	{
		print_optimization_passes();
		exit(0);
	}
	std::vector<std::string> result;
	std::stringstream ss (opt);
	std::string item;
	while (getline (ss, item, ',')) {
		if( item == "unionize" )
		{
			LOG(DEBUG) << "Enabling 'Unionize tensors' optimization pass" << std::endl;
			options.opt_unionize=true;
		}
		else if( item == "fold_casts" )
		{
			LOG(DEBUG) << "Enabling 'Fold casts' optimization pass" << std::endl;
			options.opt_fold_casts=true;
		}
		else {
			LOG(WARNING) << "Optimization pass " << item << " does not exist" << std::endl;
		}
	}
	LOG(TRACE) << "That was all optimizations" << std::endl;
}

void parse_cmdline_options(int argc, const char *argv[])
{
	args::ArgumentParser parser("Generate C code from an ONNX graph file.");
	args::Flag avr(parser, "avr", "Target AVR-GCC", {'a', "avr"});
	args::Flag noGlobals(parser, "no-globals", "Do not generate global tensors", {'n', "no-globals"});
	args::Flag externInit(parser, "extern-init", "Declare initialized tensors as extern globals", {'e', "extern-init"});
	args::Flag onlyInit(parser, "only-init", "Only generate initialized tensors (for use with --extern-init)", {'i', "only-init"});
	args::ValueFlagList<std::string> define(parser, "dim:size", "Define graph input dimension. Can be given multiple times", {'d', "define"});
	args::ValueFlag<int> loglevel(parser, "level", "Logging verbosity. 0(none)-4(all)", {'l',"log"});
	args::ValueFlag<std::string> optimizations(parser, "opt[,opt]...", "Specify optimization passes to run. ('help' to list available)", {'p', "optimizations"});
	args::Flag help(parser, "help", "Print this help text.", {'h',"help"});
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

	if (avr) { options.target_avr = true; }
	if (noGlobals) { options.no_globals = true; }
	if (externInit) { options.extern_init = true; }
	if (onlyInit) { options.only_init = true; }
	if (define) {
		for (const auto &d: args::get(define)) {
			store_define_option(d);
		}
	}
	if (optimizations) { store_optimization_passes( args::get(optimizations) ); }
	if (input) { options.input_file = args::get(input); }
	if (options.input_file == "" ) { std::cerr << "No input file given"; hint_at_help_and_exit(); }

	options.command_line_args.reserve(argc);
	for (int i = 0; i < argc; i++)
		options.command_line_args.emplace_back(argv[i]);
}

