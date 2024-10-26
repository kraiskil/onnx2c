#include "input.h"
#include "math.h"
#include <stdint.h>

extern int64_t resultLabel[1];
extern float result[1][OUTPUT_SIZE];

void entry(float tensor_conv2d_input[1][5][5][NUM_CHAN], int64_t tensor_label[1], float tensor_flatten[1][OUTPUT_SIZE]);


int main(void)
{
	int64_t outputLabel[1] = {-1};
	float output[1][OUTPUT_SIZE];

	entry(input, outputLabel, output);

	for( int i=0; i<OUTPUT_SIZE; i++)
	{
		if( isnan(result[0][i]) )
			return 1;
		if( fabs(result[0][i] - output[0][i]) > 1e-4 )
			return 1;
	}

	if( resultLabel[0] != outputLabel[0] )
		return 1;

	return 0;

}

