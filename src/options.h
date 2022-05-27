/* This file is part of onnx2c.
 *
 * Onnx2c code gereration options.
 */

#include <map>
#include <string>

struct onnx2c_opts
{
	bool quantize=false;
	bool target_avr=false;
	int logging_level=1;
	std::string input_file;
	std::map<std::string, uint32_t> dim_defines;
};

extern struct onnx2c_opts options;

/* Parse command line, fill the global 'options' struct
 * with the results
 */ 
void parse_cmdline_options(int argc, const char *argv[]);

