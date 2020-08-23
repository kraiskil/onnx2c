#include <math.h>
#include <stdio.h>
#include "test_input.h"


void entry(float tensor_flatten_input[1][130][13], float tensor_dense_3[1][10]);

float output[1][10];

#ifdef LESSON_14
#define result result_lesson14
#endif



int main(void)
{

	for( int b=0; b<10; b++) {
		entry(&test_data[b], output);

		for( int i=0; i<10; i++ ) {
			if( fabs( output[0][i] - result[b][i] ) > 0.0001 )
				return 1;
			if( isnan( output[0][i] ))
				return 1;
		}
	}
	return 0;
}

