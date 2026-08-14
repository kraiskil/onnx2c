#include "input.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPSILON 1e-6f

static int failures = 0;

#ifdef TEST_conv_simple
extern void entry(const float tensor_input[1][1][5][5], float tensor_output[1][1][4][4]);
float tensor_output[1][1][4][4];
#endif

#ifdef TEST_conv_stride2
extern void entry(const float tensor_input[1][1][7][7], float tensor_output[1][1][3][3]);
float tensor_output[1][1][3][3];
#endif

#ifdef TEST_depthwise_conv
extern void entry(const float tensor_input[1][2][4][4], float tensor_output[1][2][3][3]);
float tensor_output[1][2][3][3];
#endif

#ifdef TEST_pointwise_conv
extern void entry(const float tensor_input[1][3][4][4], float tensor_output[1][5][4][4]);
float tensor_output[1][5][4][4];
#endif

#ifdef TEST_multi_channel_conv
extern void entry(const float tensor_input[1][3][4][4], float tensor_output[1][4][2][2]);
float tensor_output[1][4][2][2];
#endif

int main(int argc, char** argv)
{
	printf("=== Im2Col smoke test ===\n");
	printf("Running generated code...\n");

#ifdef TEST_conv_simple
	entry(input_conv_simple, tensor_output);
	printf("Output[0][0][0][0] = %f\n", tensor_output[0][0][0][0]);
	float expected = 16.0f;
	printf("Expected first value to be %f\n", expected);
	if (fabsf(tensor_output[0][0][0][0] - expected) < EPSILON) {
		printf("PASSED\n");
	}
	else {
		printf("FAILED\n");
		failures++;
	}
#endif

#ifdef TEST_conv_stride2
	entry(input_conv_stride2, tensor_output);
	printf("Output[0][0][0][0] = %f\n", tensor_output[0][0][0][0]);
	float expected = 23.0f;
	printf("Expected first value to be %f\n", expected);
	if (fabsf(tensor_output[0][0][0][0] - expected) < EPSILON) {
		printf("PASSED\n");
	}
	else {
		printf("FAILED\n");
		failures++;
	}
#endif

#ifdef TEST_depthwise_conv
	entry(input_depthwise, tensor_output);
	printf("Output[0][0][0][0] = %f\n", tensor_output[0][0][0][0]);
	float expected = 14.0f;
	printf("Expected first value to be %f\n", expected);
	if (fabsf(tensor_output[0][0][0][0] - expected) < EPSILON) {
		printf("PASSED\n");
	}
	else {
		printf("FAILED\n");
		failures++;
	}
#endif

#ifdef TEST_pointwise_conv
	entry(input_pointwise, tensor_output);
	printf("Output[0][0][0][0] = %f\n", tensor_output[0][0][0][0]);
	float expected = 6.0f;
	printf("Expected first value to be %f\n", expected);
	if (fabsf(tensor_output[0][0][0][0] - expected) < EPSILON) {
		printf("PASSED\n");
	}
	else {
		printf("FAILED\n");
		failures++;
	}
#endif

#ifdef TEST_multi_channel_conv
	entry(input_multi_channel, tensor_output);
	printf("Output[0][0][0][0] = %f\n", tensor_output[0][0][0][0]);
	float expected = 975.0f;
	printf("Expected first value to be %f\n", expected);
	if (fabsf(tensor_output[0][0][0][0] - expected) < EPSILON) {
		printf("PASSED\n");
	}
	else {
		printf("FAILED\n");
		failures++;
	}
#endif

	return failures;
}
