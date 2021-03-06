#include "input.h"
#include "math.h"

extern float result[1][OUTPUT_SIZE];

void entry(float tensor_conv2d_input[1][5][5][NUM_CHAN], float tensor_flatten[1][OUTPUT_SIZE]);


int main(void)
{
	float output[1][OUTPUT_SIZE];
	//entry((float (*)[5][5][NUM_CHAN])input, output);
	entry(input, output);

	for( int i=0; i<OUTPUT_SIZE; i++)
	{
		if( isnan(result[0][i]) )
			return 1;
		if( fabs(result[0][i] - output[0][i]) > 1e-4 )
			return 1;
	}
	return 0;

}

