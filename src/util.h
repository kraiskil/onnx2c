#pragma once
#include <string>
#include <vector>
#include "onnx.pb.h"

/* ONNX names are not valid C - make name acceptable to the C compiler*/
std::string cify_name(const std::string &in);

/* Helper functions to parse attributes in a onnx NodeProto */
int parse_attribute_int(const onnx::AttributeProto &a);
std::vector<int> parse_attribute_ints(const onnx::AttributeProto &a);
float parse_attribute_float(const onnx::AttributeProto &a);
std::vector<float> parse_attribute_floats(const onnx::AttributeProto &a);
std::string parse_attribute_string(const onnx::AttributeProto &a);
std::vector<std::string> parse_attribute_strings(const onnx::AttributeProto &a);

