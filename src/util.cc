#include "error.h"
#include "util.h"

std::string cify_name(const std::string &in)
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


int parse_attribute_int(const onnx::AttributeProto &a)
{
	if( a.has_i() == false )
		ERROR("Not a int attribute");

	return a.i();
}

std::vector<int> parse_attribute_ints(const onnx::AttributeProto &a)
{
	if( a.ints_size() == 0 )
		ERROR("Not a floats attribute");

	std::vector<int> rv;

	for( int i : a.ints() )
		rv.push_back(i);

	return rv;
}

float parse_attribute_float(const onnx::AttributeProto &a)
{
	if( a.has_f() == false )
		ERROR("Not a float attribute");

	return a.f();
}
std::vector<float> parse_attribute_floats(const onnx::AttributeProto &a)
{
	if( a.floats_size() == 0 )
		ERROR("Not a floats attribute");

	std::vector<float> rv;

	for( float f : a.floats() )
		rv.push_back(f);

	return rv;
}

std::string parse_attribute_string(const onnx::AttributeProto &a)
{
	if( a.has_s() == false )
		ERROR("Not a string attribute");

	return a.s();
}

std::vector<std::string> parse_attribute_strings(const onnx::AttributeProto &a)
{
	if( a.strings_size() == 0 )
		ERROR("Not a floats attribute");

	std::vector<std::string> rv;

	for( auto s : a.strings() )
		rv.push_back(s);

	return rv;


}

