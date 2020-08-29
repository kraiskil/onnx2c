#include <math.h>
#include <stdio.h>

void entry(float tensor_dense_input[1][2], float tensor_dense_1[1][1]);


int main(void)
{
	float input1[1][2] = {{ 0.1, 0.2 }};
	float input2[1][2] = {{ 0.2, 0.2 }};
	float output1 = 0.2859681248664856;
	float output2 = 0.38861969113349915;
	float result[1][1];

	entry( input1, result );
	if( isnan(result[0][0]) )
		return 1;
	if( fabs( result[0][0] - output1 ) > 1e-5 )
		return 1;

	entry( input2, result );
	if( isnan(result[0][0]) )
		return 1;
	if( fabs( result[0][0] - output2 ) > 1e-5 )
		return 1;


	return 0;
}

