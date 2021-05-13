/* "unit test" file for checking the
 * accuracy of the generated floating point network. This should be
 * the same as in pythorch, but print it out here for completeness.
 * This loads the MNIST test set, and prints statistics.
 */
#include <assert.h>
#include <stdio.h>
#include "load.h"

void entry(float tensor_network_input[1][1][IMG_H][IMG_W], float tensor_network_output[1][10]);

int max_val(float res[1][10])
{
	float max = res[0][0];
	int idx = 0;

	for( int i=1; i<10; i++ )
	{
		if( res[0][i] > max )
		{
			idx = i;
			max = res[0][i];
		}
	}
	return idx;
}

int main(void)
{
	int label=0;
	float input[1][1][IMG_H][IMG_W];
	float result[1][10];
	int correct=0;
	int wrong=0;
	int correct_bins[10]={0};
	int wrong_bins[10]={0};

	open_files();

	for(int i=0; i<NUM_TESTS; i++)
	{
		get_char_float( i, (char_float*)&input, &label);
		assert( label >= 0 && label <= 10 );

		entry(input, result);

		int r = max_val( result );

		if( label == r ){
			correct++;
			correct_bins[label]++;
		}
		else {
			wrong++;
			wrong_bins[label]++;
		}
	}

	printf("Overall accuracy: %d / %d  = %.1f accuracy\n", correct, correct+wrong, (float)(correct*100.0)/(correct+wrong));
	for( int i=0; i<10; i++ ) {
		printf("\tAccuracy for label %d: %f%%\n", i, (100.0 * correct_bins[i])/(correct_bins[i]+wrong_bins[i]));
	}
}

