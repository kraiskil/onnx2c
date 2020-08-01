#pragma once
#include <string>
/* ONNX names are not valid C. C-ify them */
static std::string cify_name(const std::string &in)
{
	// Replace all non-allowed characters with underscore
	std::string rv = in;
	for( auto &c : rv )
	{
		// TODO: Check what the C standard allows. isalnum takes locales into account
		if( isalnum(c) )
			continue;
		c='_';
	}

	return rv;
}

