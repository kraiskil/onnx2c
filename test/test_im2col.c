/* Test harness for comparing regular conv vs im2col conv
 * Usage: gcc test_im2col.c -lm && ./test_im2col
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CMP_TOLERANCE 1e-6

/* Test case 1: Simple conv 2kernels */
static float input_1[1][1][5][5] = {{{{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}, {16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}}}};
static float kernel_1[2][1][2][2] = {
    {{{1, 1}, {1, 1}}},
    {{{1, 1}, {1, 1}}}};
static float bias_1[2] = {0, 0};

/* Regular conv node */
void node_conv2d_regular(const float input[1][1][5][5], const float kernel[2][1][2][2], const float bias[2], float output[1][2][4][4])
{
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 2; m++) {
			for (int o0 = 0; o0 < 4; o0++) {
				for (int o1 = 0; o1 < 4; o1++) {
					float sum = bias[m];
					for (int c = 0; c < 1; c++) {
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

/* Flatten output for comparison */
void flatten(float in[1][2][4][4], float out[32])
{
	for (int b = 0; b < 1; b++)
		for (int m = 0; m < 2; m++)
			for (int h = 0; h < 4; h++)
				for (int w = 0; w < 4; w++)
					out[b * 16 + m * 4 + h * 4 + w] = in[b][m][h][w];
}

int compare_float(float* a, float* b, int n)
{
	int diffs = 0;
	for (int i = 0; i < n; i++) {
		if (fabsf(a[i] - b[i]) > CMP_TOLERANCE) diffs++;
	}
	return diffs;
}

void print_output(float* t, int n)
{
	for (int i = 0; i < n; i++) {
		printf("%8.2f ", t[i]);
		if ((i + 1) % 8 == 0) printf("\n");
	}
	printf("\n");
}

int main(int argc, char* argv[])
{
	float out_reg[32], out_test[32];
	float reg_output[1][2][4][4];
	int errors;

	printf("=== Test 1: Simple Conv 2x2 kernel ===\n\n");

	/* Run regular conv */
	printf("Running regular conv...\n");
	node_conv2d_regular(input_1, kernel_1, bias_1, reg_output);
	flatten(reg_output, out_reg);

	/* Manual im2col equivalent:
	 * X: (1,1,5,5) -> im2col: (1,4,16) where 4=1*2*2, 16=4*4
	 */
	printf("Running im2col-equivalent...\n");

	/* Reconstruct im2col manually to verify */
	float im2col_x[1][4][16];
	float im2col_w[2][4];
	float mm_out[1][2][16];
	float im2col_y[1][2][4][4];

	/* im2col: extract patches from input */
	for (int b = 0; b < 1; b++) {
		for (int kh = 0; kh < 4; kh++) {
			for (int kw = 0; kw < 4; kw++) {
				for (int c = 0; c < 1; c++) {
					for (int khk = 0; khk < 2; khk++) {
						for (int kwk = 0; kwk < 2; kwk++) {
							int out_idx = kh * 4 + kw;
							int in_idx = khk * 2 + kwk;
							im2col_x[b][in_idx][out_idx] = input_1[b][c][kh + khk][kw + kwk];
						}
					}
				}
			}
		}
	}

	/* Flatten kernel to 2D matrix */
	for (int m = 0; m < 2; m++) {
		for (int i = 0; i < 4; i++) {
			int kh = i / 2;
			int kw = i % 2;
			im2col_w[m][i] = kernel_1[m][0][kh][kw];
		}
	}

	/* Matrix multiplication */
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 2; m++) {
			for (int o = 0; o < 16; o++) {
				float sum = 0;
				for (int k = 0; k < 4; k++) {
					sum += im2col_x[b][k][o] * im2col_w[m][k];
				}
				mm_out[b][m][o] = sum + bias_1[m];
			}
		}
	}

	/* Unflatten */
	for (int b = 0; b < 1; b++) {
		for (int m = 0; m < 2; m++) {
			for (int h = 0; h < 4; h++) {
				for (int w = 0; w < 4; w++) {
					im2col_y[b][m][h][w] = mm_out[b][m][h * 4 + w];
				}
			}
		}
	}

	flatten(im2col_y, out_test);

	printf("\nRegular conv output:\n");
	print_output(out_reg, 16);

	printf("\nim2col output:\n");
	print_output(out_test, 16);

	errors = compare_float(out_reg, out_test, 16);
	printf("\nDifferences: %d\n", errors);

	if (errors == 0) {
		printf("\n*** TEST PASSED: bit-accurate ***\n");
		return 0;
	}
	else {
		printf("\n*** TEST FAILED ***\n");
		return 1;
	}
}