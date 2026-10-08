/* This file is part of onnx2c.
 *
 * LinearClassifier node (ai.onnx.ml domain).
 */
#include "node.h"
#include "util.h"

#include <iomanip>
#include <sstream>
#include <unordered_map>

namespace toC {

class LinearClassifier : public Node {
	public:
	LinearClassifier()
	{
		op_name = "LinearClassifier";
		multi_class = 0;
		post_transform = "NONE";
		num_examples = 0;
		num_features = 0;
		num_classes = 0;
		binary = false;
	}

	std::vector<float> coefficients;
	std::vector<float> intercepts;
	std::vector<int64_t> classlabels_ints;
	int64_t multi_class;
	std::string post_transform;

	// Resolved input geometry, filled in by resolve()
	int num_examples;
	int num_features;
	int num_classes;
	// single intercept + 2 class labels == the binary special case
	bool binary;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes(onnx::NodeProto& node) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream& dst) const override;
};

void LinearClassifier::parseAttributes(onnx::NodeProto& node)
{
	std::unordered_map<std::string, onnx::AttributeProto> attrs;
	for (const auto& a : node.attribute()) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		attrs[a.name()] = a;
	}

	for (const auto& a : node.attribute()) {
		if (a.name() != "coefficients" && a.name() != "intercepts" &&
		    a.name() != "classlabels_ints" && a.name() != "classlabels_strings" &&
		    a.name() != "multi_class" && a.name() != "post_transform")
			ERROR("LinearClassifier " << onnx_name << ": unknown attribute " << a.name());
	}

	if (attrs.contains("classlabels_strings"))
		ERROR("LinearClassifier " << onnx_name << ": classlabels_strings (string labels) is not supported, use classlabels_ints");
	if (!attrs.contains("classlabels_ints"))
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute classlabels_ints");
	if (!attrs.contains("coefficients"))
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute coefficients");
	if (!attrs.contains("intercepts"))
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute intercepts");

	coefficients = parse_attribute_floats(attrs["coefficients"]);
	intercepts = parse_attribute_floats(attrs["intercepts"]);
	classlabels_ints = parse_attribute_ints(attrs["classlabels_ints"]);

	if (attrs.contains("multi_class"))
		multi_class = parse_attribute_int(attrs["multi_class"]);

	if (attrs.contains("post_transform"))
		post_transform = parse_attribute_string(attrs["post_transform"]);
	if (post_transform != "NONE" && post_transform != "LOGISTIC" &&
	    post_transform != "SOFTMAX" && post_transform != "SOFTMAX_ZERO")
		ERROR("LinearClassifier " << onnx_name << ": unsupported post_transform " << post_transform
		                          << ", supported values are NONE, LOGISTIC, SOFTMAX and SOFTMAX_ZERO");
}

/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void LinearClassifier::resolve(void)
{
	name_input(0, "X");
	const Tensor* X = get_input_tensor(0);

	// parseAttributes is only called when the node has attributes,
	// so re-check the required ones here.
	if (coefficients.empty())
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute coefficients");
	if (intercepts.empty())
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute intercepts");
	if (classlabels_ints.empty())
		ERROR("LinearClassifier " << onnx_name << ": missing required attribute classlabels_ints");

	if (X->data_type != onnx::TensorProto_DataType_FLOAT &&
	    X->data_type != onnx::TensorProto_DataType_DOUBLE &&
	    X->data_type != onnx::TensorProto_DataType_INT32 &&
	    X->data_type != onnx::TensorProto_DataType_INT64)
		ERROR("LinearClassifier " << onnx_name << ": unsupported input data type " << X->data_type);

	if (X->rank() == 1) {
		num_examples = 1;
		num_features = X->data_dim[0];
	}
	else if (X->rank() == 2) {
		num_examples = X->data_dim[0];
		num_features = X->data_dim[1];
	}
	else
		ERROR("LinearClassifier " << onnx_name << ": input must be of shape [N,C] or [C]");

	num_classes = static_cast<int>(intercepts.size());
	if (coefficients.size() != static_cast<size_t>(num_classes) * num_features)
		ERROR("LinearClassifier " << onnx_name << ": coefficients size " << coefficients.size()
		                          << " does not match intercepts size " << num_classes
		                          << " times number of input features " << num_features);

	if (num_classes == 1) {
		if (classlabels_ints.size() != 2)
			ERROR("LinearClassifier " << onnx_name << ": single intercept (binary classification) requires exactly 2 classlabels_ints");
		binary = true;
	}
	else {
		if (classlabels_ints.size() < static_cast<size_t>(num_classes))
			ERROR("LinearClassifier " << onnx_name << ": classlabels_ints has " << classlabels_ints.size()
			                          << " entries but intercepts defines " << num_classes << " classes");
		binary = false;
	}

	// If outputs aren't registered in the same order
	// as the model outputs are listed they will be reversed.
	Tensor* label_tensor = new Tensor;
	label_tensor->data_dim.push_back(num_examples);
	label_tensor->data_type = onnx::TensorProto_DataType_INT64;

	Tensor* scores_tensor = new Tensor;
	scores_tensor->data_dim.push_back(num_examples);
	scores_tensor->data_dim.push_back(binary ? 2 : num_classes);
	scores_tensor->data_type = onnx::TensorProto_DataType_FLOAT;

	register_output(label_tensor, "Y");
	register_output(scores_tensor, "Z");
}

/* Body of the node implementing function */
void LinearClassifier::print(std::ostream& dst) const
{
	bool print_label = is_output_N_used(0);
	bool print_scores = is_output_N_used(1);

	if (!print_label && !print_scores)
		return;

	auto float_literal = [](float v) {
		std::ostringstream str;
		str << std::setprecision(9) << v;
		std::string s = str.str();
		if (s.find('.') == std::string::npos && s.find('e') == std::string::npos &&
		    s.find('E') == std::string::npos)
			s += ".0";
		return s + "f";
	};

	INDT_1 << "/*LinearClassifier*/" << std::endl;
	INDT_1 << "static const float lc_coefficients[" << coefficients.size() << "] = {";
	for (size_t i = 0; i < coefficients.size(); i++) {
		if (i != 0)
			dst << ", ";
		dst << float_literal(coefficients[i]);
	}
	dst << "};" << std::endl;
	INDT_1 << "static const float lc_intercepts[" << intercepts.size() << "] = {";
	for (size_t i = 0; i < intercepts.size(); i++) {
		if (i != 0)
			dst << ", ";
		dst << float_literal(intercepts[i]);
	}
	dst << "};" << std::endl;
	if (print_label) {
		INDT_1 << "static const int64_t lc_labels[" << classlabels_ints.size() << "] = {";
		for (size_t i = 0; i < classlabels_ints.size(); i++) {
			if (i != 0)
				dst << ", ";
			dst << classlabels_ints[i];
		}
		dst << "};" << std::endl;
	}

	std::string x_idx = get_input_tensor(0)->rank() == 1 ? "X[f]" : "X[n][f]";

	if (binary) {
		INDT_1 << "for( uint32_t n=0; n<" << num_examples << "; n++ ) {" << std::endl;
		INDT_2 << "float lc_score = lc_intercepts[0];" << std::endl;
		INDT_2 << "for( uint32_t f=0; f<" << num_features << "; f++ ) {" << std::endl;
		INDT_3 << "lc_score += (float)" << x_idx << " * lc_coefficients[f];" << std::endl;
		INDT_2 << "}" << std::endl;
		if (print_label) {
			INDT_2 << "Y[n] = lc_score > 0 ? lc_labels[1] : lc_labels[0];" << std::endl;
		}
		if (print_scores) {
			if (post_transform == "LOGISTIC") {
				INDT_2 << "Z[n][0] = 1.0f / (1.0f + expf(lc_score));" << std::endl;
				INDT_2 << "Z[n][1] = 1.0f / (1.0f + expf(-lc_score));" << std::endl;
			}
			else {
				INDT_2 << "Z[n][0] = 1.0f - lc_score;" << std::endl;
				INDT_2 << "Z[n][1] = lc_score;" << std::endl;
			}
		}
		INDT_1 << "}" << std::endl;
		return;
	}

	INDT_1 << "for( uint32_t n=0; n<" << num_examples << "; n++ ) {" << std::endl;
	INDT_2 << "float lc_scores[" << num_classes << "];" << std::endl;
	INDT_2 << "for( int c=0; c<" << num_classes << "; c++ ) {" << std::endl;
	INDT_3 << "float lc_score = lc_intercepts[c];" << std::endl;
	INDT_3 << "for( uint32_t f=0; f<" << num_features << "; f++ ) {" << std::endl;
	INDT_4 << "lc_score += (float)" << x_idx << " * lc_coefficients[c*" << num_features << "+f];" << std::endl;
	INDT_3 << "}" << std::endl;
	INDT_3 << "lc_scores[c] = lc_score;" << std::endl;
	INDT_2 << "}" << std::endl;

	if (print_label) {
		INDT_2 << "int lc_best = 0;" << std::endl;
		INDT_2 << "for( int c=1; c<" << num_classes << "; c++ ) {" << std::endl;
		INDT_3 << "if( lc_scores[c] > lc_scores[lc_best] )" << std::endl;
		INDT_4 << "lc_best = c;" << std::endl;
		INDT_2 << "}" << std::endl;
		INDT_2 << "Y[n] = lc_labels[lc_best];" << std::endl;
	}

	if (print_scores && post_transform != "NONE") {
		if (post_transform == "LOGISTIC") {
			INDT_2 << "for( int c=0; c<" << num_classes << "; c++ )" << std::endl;
			INDT_3 << "lc_scores[c] = 1.0f / (1.0f + expf(-lc_scores[c]));" << std::endl;
		}
		else if (post_transform == "SOFTMAX" || post_transform == "SOFTMAX_ZERO") {
			INDT_2 << "float lc_max = lc_scores[0];" << std::endl;
			INDT_2 << "for( int c=1; c<" << num_classes << "; c++ ) {" << std::endl;
			INDT_3 << "if( lc_scores[c] > lc_max )" << std::endl;
			INDT_4 << "lc_max = lc_scores[c];" << std::endl;
			INDT_2 << "}" << std::endl;
			INDT_2 << "float lc_sum = 0.0f;" << std::endl;
			if (post_transform == "SOFTMAX_ZERO") {
				INDT_2 << "for( int c=0; c<" << num_classes << "; c++ ) {" << std::endl;
				INDT_3 << "if( lc_scores[c] > 0.0000001f || lc_scores[c] < -0.0000001f ) {" << std::endl;
				INDT_4 << "lc_scores[c] = expf(lc_scores[c] - lc_max);" << std::endl;
				INDT_4 << "lc_sum += lc_scores[c];" << std::endl;
				INDT_3 << "} else {" << std::endl;
				INDT_4 << "lc_scores[c] = lc_scores[c] * expf(-lc_max);" << std::endl;
				INDT_3 << "}" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			else {
				INDT_2 << "for( int c=0; c<" << num_classes << "; c++ ) {" << std::endl;
				INDT_3 << "lc_scores[c] = expf(lc_scores[c] - lc_max);" << std::endl;
				INDT_3 << "lc_sum += lc_scores[c];" << std::endl;
				INDT_2 << "}" << std::endl;
			}
			INDT_2 << "for( int c=0; c<" << num_classes << "; c++ )" << std::endl;
			INDT_3 << "lc_scores[c] /= lc_sum;" << std::endl;
		}
	}

	if (print_scores) {
		INDT_2 << "for( int c=0; c<" << num_classes << "; c++ )" << std::endl;
		INDT_3 << "Z[n][c] = lc_scores[c];" << std::endl;
	}
	INDT_1 << "}" << std::endl;
}

} // namespace toC
