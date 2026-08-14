#include "im2col.h"

#include "../tensor.h"
#include <iostream>
#include <sstream>

namespace toC {

Im2Col::Im2Col()
{
	op_name = "Im2Col";
}

std::string Im2Col::c_name(void) const
{
	return "node_" + onnx_name;
}

std::string Im2Col::c_output_decl(void)
{
	std::stringstream ss;
	const Tensor* X = get_input_tensor(0);
	const Tensor* W = get_input_tensor(arithmetic_mode == QLinearConv ? 3 : 1);
	const Tensor* Y = get_output_tensor(0);
	int64_t in_ch_per_group = in_ch / group;
	ss << "FUNC_PREFIX void " << c_name() << "(const " << X->data_type_str() << " x[" << batch << "][" << in_ch << "][" << in_h << "][" << in_w << "], ";
	ss << "const " << W->data_type_str() << " w[" << out_ch << "][" << in_ch_per_group << "][" << kernel_h << "][" << kernel_w << "]";
	if (has_bias && arithmetic_mode == Conv) {
		const Tensor* B = get_input_tensor(2);
		ss << ", const " << B->data_type_str() << " bias[" << out_ch << "]";
	}
	ss << ", " << Y->data_type_str() << " y[" << batch << "][" << out_ch << "][" << out_h << "][" << out_w << "])";
	return ss.str();
}

void Im2Col::print_node(std::ostream& dst) const
{
	if (implementation == Explicit)
		print_explicit_node(dst);
	else
		print_implicit_node(dst);
}

void Im2Col::print_implicit_node(std::ostream& dst) const
{
	dst << "\t/* Implicit/Fused Im2Col Conv */" << std::endl;

	dst << "\tfor(uint32_t b = 0; b < " << batch << "; b++) {" << std::endl;
	dst << "\t  for(uint32_t m = 0; m < " << out_ch << "; m++) {" << std::endl;
	dst << "\t    for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
	dst << "\t      for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
	print_accumulator_init(dst);
	dst << "\t        " << std::endl;

	if (group == 1) {
		dst << "\t        /* Standard convolution */" << std::endl;
		dst << "\t        for(uint32_t c = 0; c < " << in_ch << "; c++) {" << std::endl;
		dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
		dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;

		if (stride_h == 1 && stride_w == 1 && dilation_h == 1 && dilation_w == 1 && pad_h == 0 && pad_w == 0) {
			dst << "\t              int ii0 = oy + ky;" << std::endl;
			dst << "\t              int ii1 = ox + kx;" << std::endl;
			print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c][ky][kx]");
		}
		else {
			dst << "\t              int ii0 = oy * " << stride_h << " + ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
			dst << "\t              int ii1 = ox * " << stride_w << " + kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
			dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
			dst << "\t              {" << std::endl;
			print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c][ky][kx]");
			dst << "\t              }" << std::endl;
		}

		dst << "\t            }" << std::endl;
		dst << "\t          }" << std::endl;
		dst << "\t        }" << std::endl;
	}
	else {
		uint32_t goc = out_ch / group;
		uint32_t gic = in_ch / group;
		dst << "\t        /* Grouped convolution */" << std::endl;
		dst << "\t        uint32_t g = m / " << goc << ";" << std::endl;
		dst << "\t        uint32_t c_start = g * " << gic << ";" << std::endl;
		dst << "\t        for(uint32_t c = c_start; c < c_start + " << gic << "; c++) {" << std::endl;
		dst << "\t          for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
		dst << "\t            for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;
		dst << "\t              int ii0 = oy * " << stride_h << " + ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
		dst << "\t              int ii1 = ox * " << stride_w << " + kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
		dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
		dst << "\t              {" << std::endl;
		print_accumulator_calc(dst, "x[b][c][ii0][ii1]", "w[m][c - c_start][ky][kx]");
		dst << "\t              }" << std::endl;
		dst << "\t            }" << std::endl;
		dst << "\t          }" << std::endl;
		dst << "\t        }" << std::endl;
	}

	dst << std::endl;
	print_accumulator_finalize(dst);
	dst << "\t      }" << std::endl;
	dst << "\t    }" << std::endl;
	dst << "\t  }" << std::endl;
	dst << "\t}" << std::endl;
}

void Im2Col::print_explicit_node(std::ostream& dst) const
{
	if (arithmetic_mode != Conv) {
		dst << "\t/* Explicit Im2Col is only implemented for floating-point Conv; using implicit path. */" << std::endl;
		print_implicit_node(dst);
		return;
	}

	int64_t in_ch_per_group = in_ch / group;
	int64_t out_ch_per_group = out_ch / group;
	int64_t k_dim = in_ch_per_group * kernel_h * kernel_w;
	int64_t p_dim = out_h * out_w;
	const Tensor* X = get_input_tensor(0);
	const Tensor* Y = get_output_tensor(0);
	std::string x_type = X->data_type_str();
	std::string y_type = Y->data_type_str();
	int64_t elem_size = X->data_elem_size();
	int64_t workspace_bytes = k_dim * p_dim * elem_size;

	dst << "\t/* Explicit/Materialized Im2Col Conv */" << std::endl;
	dst << "\t/* Layout: x_col[K][P], K=(Cin/groups)*Kh*Kw=" << k_dim
	    << ", P=Hout*Wout=" << p_dim << ". */" << std::endl;
	dst << "\t/* Peak live explicit im2col workspace: " << workspace_bytes
	    << " bytes per batch/group. */" << std::endl;
	dst << "\t#ifndef ONNX2C_C_GLOBAL" << std::endl;
	dst << "\t#ifdef __cplusplus" << std::endl;
	dst << "\t#define ONNX2C_C_GLOBAL ::" << std::endl;
	dst << "\t#else" << std::endl;
	dst << "\t#define ONNX2C_C_GLOBAL" << std::endl;
	dst << "\t#endif" << std::endl;
	dst << "\t#endif" << std::endl;
	dst << "\t" << x_type << "* x_col = (" << x_type << "*)ONNX2C_C_GLOBAL calloc((size_t)" << k_dim << " * (size_t)" << p_dim << ", sizeof(" << x_type << "));" << std::endl;
	dst << "\tif (!x_col) return;" << std::endl;
	dst << "\t#ifdef ONNX2C_EXPLICIT_IM2COL_PROFILE" << std::endl;
	dst << "\tdouble onnx2c_im2col_materialize_ms = 0.0;" << std::endl;
	dst << "\tdouble onnx2c_im2col_gemm_ms = 0.0;" << std::endl;
	dst << "\t#endif" << std::endl;
	dst << "\tfor(uint32_t b = 0; b < " << batch << "; b++) {" << std::endl;
	dst << "\t  for(uint32_t g = 0; g < " << group << "; g++) {" << std::endl;
	dst << "\t    #ifdef ONNX2C_EXPLICIT_IM2COL_PROFILE" << std::endl;
	dst << "\t    double onnx2c_phase_t0 = onnx2c_profile_now_ms();" << std::endl;
	dst << "\t    #endif" << std::endl;
	dst << "\t    for(size_t i = 0; i < (size_t)" << k_dim << " * (size_t)" << p_dim << "; i++)" << std::endl;
	dst << "\t      x_col[i] = 0;" << std::endl;
	dst << "\t    /* Materialize x_col[K][P]. */" << std::endl;
	dst << "\t    for(uint32_t c_local = 0; c_local < " << in_ch_per_group << "; c_local++) {" << std::endl;
	dst << "\t      uint32_t c = g * " << in_ch_per_group << " + c_local;" << std::endl;
	dst << "\t      for(uint32_t ky = 0; ky < " << kernel_h << "; ky++) {" << std::endl;
	dst << "\t        for(uint32_t kx = 0; kx < " << kernel_w << "; kx++) {" << std::endl;
	dst << "\t          uint32_t k = (c_local * " << kernel_h << " + ky) * " << kernel_w << " + kx;" << std::endl;
	dst << "\t          for(int32_t oy = 0; oy < " << out_h << "; oy++) {" << std::endl;
	dst << "\t            for(int32_t ox = 0; ox < " << out_w << "; ox++) {" << std::endl;
	dst << "\t              uint32_t p = (uint32_t)oy * " << out_w << " + (uint32_t)ox;" << std::endl;
	dst << "\t              int ii0 = oy * " << stride_h << " + (int32_t)ky * " << dilation_h << " - " << pad_h << ";" << std::endl;
	dst << "\t              int ii1 = ox * " << stride_w << " + (int32_t)kx * " << dilation_w << " - " << pad_w << ";" << std::endl;
	dst << "\t              if(ii0 >= 0 && ii0 < " << in_h << " && ii1 >= 0 && ii1 < " << in_w << ")" << std::endl;
	dst << "\t                x_col[(size_t)k * " << p_dim << " + p] = x[b][c][ii0][ii1];" << std::endl;
	dst << "\t            }" << std::endl;
	dst << "\t          }" << std::endl;
	dst << "\t        }" << std::endl;
	dst << "\t      }" << std::endl;
	dst << "\t    }" << std::endl;
	dst << "\t    #ifdef ONNX2C_EXPLICIT_IM2COL_PROFILE" << std::endl;
	dst << "\t    onnx2c_im2col_materialize_ms += onnx2c_profile_now_ms() - onnx2c_phase_t0;" << std::endl;
	dst << "\t    onnx2c_phase_t0 = onnx2c_profile_now_ms();" << std::endl;
	dst << "\t    #endif" << std::endl;
	dst << "\t    /* MatMul/GEMM: W_col[Cout/group][K] * x_col[K][P]. */" << std::endl;
	dst << "\t    /* Loop order keeps P contiguous in the innermost loop. */" << std::endl;
	dst << "\t    for(uint32_t m_local = 0; m_local < " << out_ch_per_group << "; m_local++) {" << std::endl;
	dst << "\t      uint32_t m = g * " << out_ch_per_group << " + m_local;" << std::endl;
	dst << "\t      " << y_type << "* y_plane = &y[b][m][0][0];" << std::endl;
	dst << "\t      for(uint32_t p = 0; p < " << p_dim << "; p++) {" << std::endl;
	dst << "\t        y_plane[p] = ";
	if (has_bias)
		dst << "bias[m]";
	else
		dst << "0";
	dst << ";" << std::endl;
	dst << "\t      }" << std::endl;
	dst << "\t    }" << std::endl;
	dst << "\t    for(uint32_t k = 0; k < " << k_dim << "; k++) {" << std::endl;
	dst << "\t      uint32_t c_local = k / (" << kernel_h << " * " << kernel_w << ");" << std::endl;
	dst << "\t      uint32_t rem = k % (" << kernel_h << " * " << kernel_w << ");" << std::endl;
	dst << "\t      uint32_t ky = rem / " << kernel_w << ";" << std::endl;
	dst << "\t      uint32_t kx = rem % " << kernel_w << ";" << std::endl;
	dst << "\t      for(uint32_t m_local = 0; m_local < " << out_ch_per_group << "; m_local++) {" << std::endl;
	dst << "\t        uint32_t m = g * " << out_ch_per_group << " + m_local;" << std::endl;
	dst << "\t        " << y_type << " weight = w[m][c_local][ky][kx];" << std::endl;
	dst << "\t        " << y_type << "* y_plane = &y[b][m][0][0];" << std::endl;
	dst << "\t        " << x_type << "* x_row = &x_col[(size_t)k * " << p_dim << "];" << std::endl;
	dst << "\t        for(uint32_t p = 0; p < " << p_dim << "; p++) {" << std::endl;
	dst << "\t          y_plane[p] += weight * x_row[p];" << std::endl;
	dst << "\t        }" << std::endl;
	dst << "\t      }" << std::endl;
	dst << "\t    }" << std::endl;
	dst << "\t    #ifdef ONNX2C_EXPLICIT_IM2COL_PROFILE" << std::endl;
	dst << "\t    onnx2c_im2col_gemm_ms += onnx2c_profile_now_ms() - onnx2c_phase_t0;" << std::endl;
	dst << "\t    #endif" << std::endl;
	dst << "\t  }" << std::endl;
	dst << "\t}" << std::endl;
	dst << "\tONNX2C_C_GLOBAL free(x_col);" << std::endl;
	dst << "\t#ifdef ONNX2C_EXPLICIT_IM2COL_PROFILE" << std::endl;
	dst << "\tONNX2C_C_GLOBAL fprintf(stderr, \"explicit_im2col_profile,%s,%lld,%lld,%lld,%lld,%lld,%.9f,%.9f,0.000000000\\n\", \"" << onnx_name
	    << "\", (long long)" << out_ch_per_group << ", (long long)" << k_dim << ", (long long)" << p_dim
	    << ", (long long)" << workspace_bytes << ", (long long)" << (out_ch_per_group * k_dim * p_dim)
	    << ", onnx2c_im2col_materialize_ms, onnx2c_im2col_gemm_ms);" << std::endl;
	dst << "\t#endif" << std::endl;
}

void Im2Col::print_accumulator_init(std::ostream& dst) const
{
	if (arithmetic_mode == QLinearConv) {
		dst << "\t        int32_t acc = ";
		if (has_bias)
			dst << "bias[m]";
		else
			dst << "0";
		dst << ";" << std::endl;
		return;
	}

	const Tensor* Y = get_output_tensor(0);
	dst << "\t        " << Y->data_type_str() << " acc = ";
	if (arithmetic_mode == Conv && has_bias)
		dst << "bias[m]";
	else
		dst << "0";
	dst << ";" << std::endl;
}

void Im2Col::print_accumulator_calc(std::ostream& dst, const std::string& x_idx, const std::string& w_idx) const
{
	if (arithmetic_mode == Conv) {
		dst << "\t                acc += " << x_idx << " * " << w_idx << ";" << std::endl;
		return;
	}

	if (arithmetic_mode == ConvInteger) {
		std::string x_zero = has_x_zero_point ? "x_zero_point[0]" : "0";
		std::string w_zero = has_w_zero_point ? "w_zero_point[0]" : "0";
		dst << "\t                acc += (" << x_idx << " - " << x_zero << ") * (" << w_idx << " - " << w_zero << ");" << std::endl;
		return;
	}

	dst << "\t                acc += ((int32_t)" << x_idx << " - x_zero_point[0]) * ((int32_t)" << w_idx << " - w_zero_point[0]);" << std::endl;
}

void Im2Col::print_accumulator_finalize(std::ostream& dst) const
{
	if (arithmetic_mode == QLinearConv) {
		const Tensor* X_scale = get_input_tensor(1);
		const Tensor* Y = get_output_tensor(0);
		std::string float_dtype = X_scale->data_type_str();
		dst << "\t        " << float_dtype << " scaled = ((" << float_dtype << ")acc) * (x_scale[0] * w_scale[0]) / y_scale[0];" << std::endl;
		dst << "\t        scaled = scaled + (" << float_dtype << ")y_zero_point[0];" << std::endl;
		dst << "\t        y[b][m][oy][ox] = (" << Y->data_type_str() << ") roundf(scaled);" << std::endl;
		return;
	}

	dst << "\t        y[b][m][oy][ox] = acc;" << std::endl;
}

void Im2Col::resolve_datatypes(void)
{
}

void Im2Col::print(std::ostream& dst) const
{
	print_node(dst);
}

} // namespace toC
