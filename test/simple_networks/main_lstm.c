#include <math.h>
#include <stdio.h>

float input[1][1][5] = 
{{{ 1, 0, 0, 0, 0 }}};

extern float reference[1][3]; 
extern int num_rounds;

void entry(float tensor_input[1][1][5], float tensor_output[1][3]);


int main(void)
{
	float output[1][3];

	for( int i=0; i<num_rounds; i++)
		entry(input, output);

	for( int i=0; i<3; i++)
	{
		if( isnan(output[0][i]) )
			return 1;
		if( fabs(output[0][i] - reference[0][i]) > 1e-5 )
			return 1;
	}
	return 0;

}

