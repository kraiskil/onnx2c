/* Numerical test comparing generated conv code vs generated im2col code */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define CLIP(X, L) (MAX(MIN(X, L), -L))

#ifndef CMP_TOLERANCE
#define CMP_TOLERANCE 1e-4
#endif

/* Test case 1: conv_k2 with stride */
static float input_k2[1][1][7][7] = {{{{1, 2, 3, 4, 5, 6, 7}},
                                      {{8, 9, 10, 11, 12, 13, 14}},
                                      {{15, 16, 17, 18, 19, 20, 21}},
                                      {{22, 23, 24, 25, 26, 27, 28}},
                                      {{29, 30, 31, 32, 33, 34, 35}},
                                      {{36, 37, 38, 39, 40, 41, 42}},
                                      {{43, 44, 45, 46, 47, 48, 49}}}};

static float kernel_k2[1][1][2][2] = {{{{1, 2}, {3, 4}}}};
static float bias_k2[1] = {0};

static float input_2ch[1][3][5][5];
static float kernel_2ch[1][3][2][2];
static float bias_2ch[1] = {0};

void init_complex_inputs()
{
	/* Multi-channel input */
	for (int b = 0; b < 1; b++)
		for (int c = 0; c < 3; c++)
			for (int h = 0; h < 5; h++)
				for (int w = 0; w < 5; w++)
					input_2ch[b][c][h][w] = (float)(c * 25 + h * 5 + w + 1);

	/* Multi-channel kernel */
	for (int m = 0; m < 1; m++)
		for (int c = 0; c < 3; c++)
			for (int kh = 0; kh < 2; kh++)
				for (int kw = 0; kw < 2; kw++)
					kernel_2ch[m][c][kh][kw] = (float)(c * 4 + kh * 2 + kw + 1) / 100.0f;
}

/* Regular conv with stride */
void conv_stride(const float input[1][1][7][7], const float kernel[1][1][2][2], const float bias[1], float output[1][1][3][3])
{
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 1; m++) {
			for (int o0 = 0, i0 = 0; o0 < 3; o0++, i0 += 2) {
				for (int o1 = 0, i1 = 0; o1 < 3; o1++, i1 += 2) {
					float sum = bias[m];
					for (int c = 0; c < 1; c++) {
						for (int k0 = 0; k0 < 2; k0++) {
							for (int k1 = 0; k1 < 2; k1++) {
								sum += input[b][c][i0 + k0][i1 + k1] * kernel[m][c][k0][k1];
							}
						}
					}
					output[b][m][o0][o1] = sum;
				}
			}
		}
	}
}

/* im2col conv with stride */
void im2col_stride(const float input[1][1][7][7], const float kernel[1][1][2][2], const float bias[1], float output[1][1][3][3])
{
	/* im2col: create column matrix */
	float im2col_x[1][4][9]; /* C*kH*kW=4, outH*outW=9 */
	float im2col_w[1][4];
	float mm_out[1][1][9];
	float im2col_y[1][1][3][3];

	/* Extract patches with stride 2 */
	int out_h = 3, out_w = 3;
	for (int b = 0; b < 1; b++) {
		for (int oh = 0; oh < out_h; oh++) {
			for (int ow = 0; ow < out_w; ow++) {
				int ih_start = oh * 2;
				int iw_start = ow * 2;

				for (int kh = 0; kh < 2; kh++) {
					for (int kw = 0; kw < 2; kw++) {
						int kin = kh * 2 + kw;
						int oout = oh * 3 + ow;
						im2col_x[b][kin][oout] = input[b][0][ih_start + kh][iw_start + kw];
					}
				}
			}
		}
	}

	/* Flatten kernel */
	for (int m = 0; m < 1; m++) {
		for (int i = 0; i < 4; i++) {
			int kh = i / 2;
			int kw = i % 2;
			im2col_w[m][i] = kernel[m][0][kh][kw];
		}
	}

	/* MatMul */
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 1; m++) {
			for (int o = 0; o < 9; o++) {
				float sum = 0;
				for (int k = 0; k < 4; k++) {
					sum += im2col_x[b][k][o] * im2col_w[m][k];
				}
				mm_out[b][m][o] = sum + bias[m];
			}
		}
	}

	/* Unflatten */
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 1; m++) {
			for (int h = 0; h < 3; h++) {
				for (int w = 0; w < 3; w++) {
					im2col_y[b][m][h][w] = mm_out[b][m][h * 3 + w];
				}
			}
		}
	}

	/* Copy to output */
	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 1; m++)
			for (int h = 0; h < 3; h++)
				for (int w = 0; w < 3; w++)
					output[b][m][h][w] = im2col_y[b][m][h][w];
}

/* Conv with multiple input channels */
void conv_multi_ch(const float input[1][3][5][5], const float kernel[1][3][2][2], const float bias[1], float output[1][1][4][4])
{
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 1; m++) {
			for (int o0 = 0; o0 < 4; o0++) {
				for (int o1 = 0; o1 < 4; o1++) {
					float sum = bias[m];
					for (int c = 0; c < 3; c++) {
						for (int k0 = 0; k0 < 2; k0++) {
							for (int k1 = 0; k1 < 2; k1++) {
								sum += input[b][c][o0 + k0][o1 + k1] * kernel[m][c][k0][k1];
							}
						}
					}
					output[b][m][o0][o1] = sum;
				}
			}
		}
	}
}

void im2col_multi_ch(const float input[1][3][5][5], const float kernel[1][3][2][2], const float bias[1], float output[1][1][4][4])
{
	float im2col_x[1][12][16]; /* C*kH*kW=12, outH*outW=16 */
	float im2col_w[1][12];
	float mm_out[1][1][16];
	float im2col_y[1][1][4][4];
	int C = 3, kH = 2, kW = 2, outH = 4, outW = 4;

	/* Extract patches */
	for (int b = 0; b < 1; b++) {
		for (int oh = 0; oh < outH; oh++) {
			for (int ow = 0; ow < outW; ow++) {
				int oout = oh * outW + ow;
				for (int c = 0; c < C; c++) {
					for (int kh = 0; kh < kH; kh++) {
						for (int kw = 0; kw < kW; kw++) {
							int kin = c * kH * kW + kh * kW + kw;
							im2col_x[b][kin][oout] = input[b][c][oh + kh][ow + kw];
						}
					}
				}
			}
		}
	}

	/* Flatten kernel */
	for (int m = 0; m < 1; m++) {
		for (int c = 0; c < C; c++) {
			for (int kh = 0; kh < kH; kh++) {
				for (int kw = 0; kw < kW; kw++) {
					int kin = c * kH * kW + kh * kW + kw;
					im2col_w[m][kin] = kernel[m][c][kh][kw];
				}
			}
		}
	}

	/* MatMul */
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 1; m++) {
			for (int o = 0; o < 16; o++) {
				float sum = 0;
				for (int k = 0; k < 12; k++) {
					sum += im2col_x[b][k][o] * im2col_w[m][k];
				}
				mm_out[b][m][o] = sum + bias[m];
			}
		}
	}

	/* Unflatten */
	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 1; m++)
			for (int h = 0; h < 4; h++)
				for (int w = 0; w < 4; w++)
					im2col_y[b][m][h][w] = mm_out[b][m][h * 4 + w];

	/* Copy */
	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 1; m++)
			for (int h = 0; h < 4; h++)
				for (int w = 0; w < 4; w++)
					output[b][m][h][w] = im2col_y[b][m][h][w];
}

int compare(float* a, float* b, int n, float tol)
{
	int diffs = 0;
	for (int i = 0; i < n; i++) {
		if (fabsf(a[i] - b[i]) > tol) diffs++;
	}
	return diffs;
}

void print_arr(float* a, int n)
{
	for (int i = 0; i < n; i++) {
		printf("%8.2f ", a[i]);
		if ((i + 1) % 8 == 0) printf("\n");
	}
	printf("\n");
}

int main()
{
	float out1_reg[9], out1_im2[9];
	float out2_reg[16], out2_im2[16];
	float reg1[1][1][3][3], im2_1[1][1][3][3];
	float reg2[1][1][4][4], im2_2[1][1][4][4];
	float arr1_reg[9], arr1_im2[9];
	float arr2_reg[16], arr2_im2[16];

	printf("=== Test 1: Conv with stride ===\n");
	conv_stride(input_k2, kernel_k2, bias_k2, reg1);
	im2col_stride(input_k2, kernel_k2, bias_k2, im2_1);

	/* Flatten */
	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 1; m++)
			for (int h = 0; h < 3; h++)
				for (int w = 0; w < 3; w++)
					arr1_reg[b * 9 + m * 9 + h * 3 + w] = reg1[b][m][h][w];

	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 1; m++)
			for (int h = 0; h < 3; h++)
				for (int w = 0; w < 3; w++)
					arr1_im2[b * 9 + m * 9 + h * 3 + w] = im2_1[b][m][h][w];

	int err1 = compare(arr1_reg, arr1_im2, 9, CMP_TOLERANCE);
	printf("Regular: ");
	print_arr(arr1_reg, 9);
	printf("im2col:  ");
	print_arr(arr1_im2, 9);
	printf("Diffs: %d\n", err1);
	printf("Result: %s\n\n", err1 == 0 ? "PASSED" : "FAILED");

	printf("=== Test 2: Multi-channel ===\n");
	init_complex_inputs();
	conv_multi_ch(input_2ch, kernel_2ch, bias_2ch, reg2);
	im2col_multi_ch(input_2ch, kernel_2ch, bias_2ch, im2_2);

	for (int i = 0; i < 16; i++)
		arr2_reg[i] = ((float*)reg2)[i];
	for (int i = 0; i < 16; i++)
		arr2_im2[i] = ((float*)im2_2)[i];

	int err2 = compare(arr2_reg, arr2_im2, 16, CMP_TOLERANCE);
	printf("Regular: ");
	print_arr(arr2_reg, 16);
	printf("im2col:  ");
	print_arr(arr2_im2, 16);
	printf("Diffs: %d\n", err2);
	printf("Result: %s\n\n", err2 == 0 ? "PASSED" : "FAILED");

	if (err1 == 0 && err2 == 0) {
		printf("=== ALL TESTS PASSED ===\n");
		return 0;
	}
	else {
		printf("=== SOME TESTS FAILED ===\n");
		return 1;
	}
}