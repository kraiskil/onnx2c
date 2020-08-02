/* Test runner for model.onnx
 * The model calculates sin(x) on the interval [0,2pi]
 */
#include <stdio.h>
#include <math.h>

/* TODO! Thes reference value are taken from onnx2c-generated code!
 *       So currently it measures only that nothing changes in the codegen
 *       not so much that it is error free. But the values do look like
 *       an outline of a sine curve: sin(i/10), so codegen can't be all wrong :) */
float reference[60] = {
0.039335, 0.125240, 0.211174, 0.297108, 0.383042, 0.468976, 0.554910, 0.640844,
0.726778, 0.812712, 0.874693, 0.919462, 0.952302, 0.978232, 1.007799, 1.037366,
1.063581, 1.065023, 1.031677, 0.978448, 0.925219, 0.871989, 0.818761, 0.765532,
0.711860, 0.619119, 0.526376, 0.433634, 0.340892, 0.248150, 0.155408, 0.062667,
-0.030076, -0.122818, -0.215559, -0.308301, -0.401044, -0.493785, -0.586527, -0.679270,
-0.772011, -0.842394, -0.861900, -0.881406, -0.900912, -0.920417, -0.939924, -0.959430,
-0.978935, -0.955388, -0.905960, -0.856532, -0.807104, -0.757676, -0.708248, -0.658821,
-0.609393, -0.536978, -0.438632, -0.340286
};

void entry(float tensor_dense_2_input[1], float tensor_dense_4[1]);

int main( void )
{
	float in, out;
	in=0;
	int rv=0;

	for( int i=0; i<60; i++)
	{
		entry(&in,&out);
		if( fabs(out - reference[i]) > 1e-5 )
		{
			printf("Wrong result: i=%d, in=%f, got=%f, expected=%f\n",
				i, in, out, reference[i]);
			rv=1;
		}
		in+=0.1; 
	}
	return rv;
}
