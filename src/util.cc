/* This file is part of onnx2c
 */

#include "error.h"
#include "options.h"
#include "tensor.h"
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

std::vector<int64_t> parse_attribute_ints(const onnx::AttributeProto &a)
{
	if( a.ints_size() == 0 )
		ERROR("Not a ints attribute");

	std::vector<int64_t> rv;

	for( auto i : a.ints() )
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

toC::Tensor* parse_attribute_tensor(const onnx::AttributeProto &a)
{
	if( a.type() != onnx::AttributeProto_AttributeType_TENSOR )
		ERROR("Attribute type is not tensor");
	if( a.has_t() == false )
		ERROR("No tensor in attribute");

	toC::Tensor *t = new toC::Tensor;
	t->parse_onnx_tensor( a.t() );
	t->initialize = true;
	t->isConst = true;
	return t;
}

std::string constant_acces_code(const std::string plain)
{
	if( !options.target_avr )
		return plain;

	std::string rv = "RD_PROGMEM(" + plain + ")";
	return rv;
}

std::string cast_to_ndim_arrayptr(const toC::Tensor *t, std::string shortname)
{
	std::string idxstr="";
	for( unsigned d=1; d<t->rank(); d++){
		idxstr += "[";
		idxstr += std::to_string(t->data_dim[d]);
		idxstr += "]";
	}

	std::string rv="";

	// float (*X)[1][2]
	rv += t->data_type_str();
	rv += "(*";
	rv += shortname;
	rv += ")";
	rv += idxstr;

	// float (*X)[1][2] = (float (*)[1][2])tensor_nodename_42;
	rv += " = (";
	rv += t->data_type_str();
	rv += "(*)";
	rv += idxstr;
	rv += ")";
	rv += t->cname();
	rv += ";";

	return rv;
}


bool isFloat(onnx::TensorProto_DataType data_type)
{
	return data_type == onnx::TensorProto_DataType_FLOAT
	     ||data_type == onnx::TensorProto_DataType_DOUBLE;
}
bool isInt(onnx::TensorProto_DataType data_type)
{
	return data_type == onnx::TensorProto_DataType_UINT8
	     ||data_type == onnx::TensorProto_DataType_INT8
	     ||data_type == onnx::TensorProto_DataType_UINT16
	     ||data_type == onnx::TensorProto_DataType_INT16
	     ||data_type == onnx::TensorProto_DataType_UINT32
	     ||data_type == onnx::TensorProto_DataType_INT32
	     ||data_type == onnx::TensorProto_DataType_UINT64
	     ||data_type == onnx::TensorProto_DataType_INT64;
}


void print_loops_over_dims(std::ostream &dst, const toC::Tensor *t, std::string prefix, unsigned indents)
{
	for( unsigned dim=0; dim<t->rank(); dim++) {
		std::string loopvar = prefix + std::to_string(dim);
		for(unsigned i=0; i<indents; i++)
			dst << "\t";
		dst << "for( uint32_t " << loopvar << "=0; ";
		dst <<       loopvar << " < " << t->data_dim[dim] << "; ";
		dst <<       loopvar << "++) {"<< std::endl;
	}
}
void print_loop_closes_over_dims(std::ostream &dst, const toC::Tensor *t, unsigned indents)
{
	for( unsigned dim=0; dim<t->rank(); dim++) {
		for(unsigned i=0; i<indents; i++)
			dst << "\t";
		dst << "}"<< std::endl;
	}
}

std::string broadcast(const toC::Tensor *t, const std::string &name, int to_rank) {
	if (t->is_scalar()) {
		return "*" + name;
	}
	std::ostringstream dst;
	dst << name;
	for (int i = 0; i < (int)t->data_dim.size(); i++) {
		if (t->data_dim[i] == 1) {
			dst << "[0]";
		} else {
			dst << "[i" << (to_rank - t->data_dim.size() + i) << "]";
		}
	}
	return dst.str();
}

