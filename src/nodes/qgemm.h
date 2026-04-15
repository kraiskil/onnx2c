/* This file is part of onnx2c.
 *
 * QGemm
 *
 * Quantized GEMM (QOperator form).
 *
 * Supported inputs:
 *   A, a_scale, a_zero_point, B, b_scale, b_zero_point, C, y_scale, y_zero_point
 * with scalar scales/zero-points for A/B/Y and int32 bias C (broadcastable).
 */

#pragma once

#include "node.h"
#include <cmath>

namespace toC {

class QGemm : public Node {
	public:
	QGemm()
	{
		op_name = "QGemm";
		alpha = 1.0f;
		transA = 0;
		transB = 0;
	}

	float alpha;
	int transA;
	int transB;

	void name_scalar_input(unsigned input_no, std::string name)
	{
		name_input(input_no, name);
		const Tensor* t = get_input_tensor(input_no);
		if (!(t->data_dim.size() == 0 || (t->data_dim.size() == 1 && t->data_dim[0] == 1))) {
			ERROR(name << " must be scalar");
		}
	}

	void parseAttributes(onnx::NodeProto& node) override
	{
		for (const auto& a : node.attribute()) {
			if (a.name() == "alpha")
				alpha = parse_attribute_float(a);
			else if (a.name() == "transA")
				transA = parse_attribute_int(a);
			else if (a.name() == "transB")
				transB = parse_attribute_int(a);
		}
	}

	void resolve(void) override
	{
		// Inputs:
		// 0 A
		// 1 a_scale
		// 2 a_zero_point
		// 3 B
		// 4 b_scale
		// 5 b_zero_point
		// 6 C (bias, int32, broadcastable)
		// 7 y_scale
		// 8 y_zero_point
		name_input(0, "A");
		name_scalar_input(1, "a_scale");
		name_scalar_input(2, "a_zero_point");
		name_input(3, "B");
		name_scalar_input(4, "b_scale");
		name_scalar_input(5, "b_zero_point");
		name_input(6, "C");
		name_scalar_input(7, "y_scale");
		name_scalar_input(8, "y_zero_point");

		if (transA != 0)
			ERROR("Unimplemented: QGemm transA");
		if (alpha != 1.0f)
			ERROR("Unimplemented: QGemm alpha != 1");

		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(3);
		const Tensor* y_zero_point = get_input_tensor(8);

		if (A->rank() != 2 || B->rank() != 2)
			ERROR("Unimplemented: QGemm only supports rank-2 matrices");

		int M = A->data_dim[0];
		int K = A->data_dim[1];
		int N = transB ? B->data_dim[0] : B->data_dim[1];
		int Kb = transB ? B->data_dim[1] : B->data_dim[0];
		if (Kb != K)
			ERROR("Reduction dimension mismatch in QGemm");

		Tensor* Y = new Tensor;
		Y->data_dim = {M, N};
		Y->data_type = y_zero_point->data_type;
		register_output(Y, "Y");
	}

	void print(std::ostream& dst) const override
	{
		const Tensor* A = get_input_tensor(0);
		const Tensor* B = get_input_tensor(3);
		const Tensor* C = get_input_tensor(6);
		const Tensor* Y = get_output_tensor(0);

		int M = A->data_dim[0];
		int K = A->data_dim[1];
		int N = transB ? B->data_dim[0] : B->data_dim[1];

		int C0 = 1;
		int C1 = 1;
		if (C->is_scalar()) {
			C0 = C1 = 1;
		}
		else if (C->rank() == 1) {
			int dim = C->data_dim[0];
			if (dim == M) {
				C0 = M;
				C1 = 1;
			}
			else if (dim == N) {
				C0 = 1;
				C1 = N;
			}
			else if (dim == 1) {
				C0 = 1;
				C1 = 1;
			}
			else {
				ERROR("C dimension mismatch in QGemm");
			}
		}
		else if (C->rank() == 2) {
			C0 = C->data_dim[0];
			C1 = C->data_dim[1];
		}
		else {
			ERROR("C has too many dimensions in QGemm");
		}

		std::string C_idx;
		C_idx += (C0 <= 1) ? "[0]" : "[r]";
		C_idx += (C1 <= 1) ? "[0]" : "[c]";

		INDT_1 << "/* QGemm */" << std::endl;
		INDT_1 << "const int32_t (*C_)[" << C1 << "] = (const int32_t(*)[" << C1 << "])C;" << std::endl;

		auto [lower, upper] = Y->get_type_bounds();
		std::string float_dtype = get_input_tensor(1)->data_type_str();

		INDT_1 << "for( uint32_t r=0; r<" << M << "; r++ ) {" << std::endl;
		INDT_2 << "for( uint32_t c=0; c<" << N << "; c++ ) {" << std::endl;
		INDT_3 << "int32_t acc32 = C_" << C_idx << ";" << std::endl;
		INDT_3 << "for( uint32_t i=0; i<" << K << "; i++ ) {" << std::endl;
		std::string B_el = transB ? "B[c][i]" : "B[i][c]";
		INDT_4 << "acc32 += ((int32_t)A[r][i] - (int32_t)a_zero_point[0]) * ((int32_t)" << B_el
		       << " - (int32_t)b_zero_point[0]);" << std::endl;
		INDT_3 << "}" << std::endl;
		INDT_3 << float_dtype << " scale = (" << float_dtype << ") (a_scale[0] * b_scale[0]) / y_scale[0];" << std::endl;
		INDT_3 << "double scaled = ((double) acc32) * (double) scale;" << std::endl;
		INDT_3 << "scaled = scaled + (double) y_zero_point[0];" << std::endl;
		INDT_3 << "int t = (int) llround(scaled);" << std::endl;
		INDT_3 << "t = MIN(MAX(t, " << lower << "), " << upper << ");" << std::endl;
		INDT_3 << "Y[r][c] = (" << Y->data_type_str() << ") t;" << std::endl;
		INDT_2 << "}" << std::endl;
		INDT_1 << "}" << std::endl;
	}
};

} // namespace toC
