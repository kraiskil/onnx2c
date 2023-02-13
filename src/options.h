/* This file is part of onnx2c.
 *
 * Onnx2c code gereration options.
 */

#pragma once
#include <map>
#include <string>
#include <stdint.h>

struct onnx2c_opts
{
	bool quantize=false;
	bool target_avr=false;
	bool opt_unionize=true;
	bool opt_fold_casts=true;
	/*
	 * logging levels are
	 * cmd line     aixlog     Use
	 * -------------------------------
	 *   -          FATAL
	 *   0          ERROR      Bad input or missing feature in onnx2c.
	 *   1          WARNING    Valid input, but onnx2c output might not be conformant.
	 *   2          INFO       Generic info, warnigns about suboptimal input.
	 *   3          DEBUG      Notes on generated nodes and tensors.
	 *   4          TRACE      Detailed info on generated nodes and tensors.
	 */
	#ifndef DEFAULT_LOG_LEVEL
	#define DEFAULT_LOG_LEVEL 2
	#endif
	int logging_level=DEFAULT_LOG_LEVEL;  // Default level set by CMake. 1 in release, 4 in debug builds
	std::string input_file;
	std::map<std::string, uint32_t> dim_defines;
};

extern struct onnx2c_opts options;

/* Parse command line, fill the global 'options' struct
 * with the results
 */ 
void parse_cmdline_options(int argc, const char *argv[]);

