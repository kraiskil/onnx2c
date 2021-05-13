/* "unit test" file for checking the
 * accuracy of the generated quantized network.
 * This loads the MNIST test set, and prints statistics.
 */
#include <stdio.h>
#include "load.h"

void entry(int8_t tensor_network_input[1][1][IMG_H][IMG_W], int8_t tensor_network_output[1][10]);

int index_of_max(int8_t res[1][10])
{
	int8_t max = res[0][0];
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
	int label;
	int8_t input[1][1][IMG_H][IMG_W];
	uint8_t raw_input[1][IMAGE_SIZE];
	int8_t result[1][10];
	int correct=0;
	int wrong=0;
	int correct_bins[10]={0};
	int wrong_bins[10]={0};

	open_files();

	for(int i=0; i<NUM_TESTS; i++)
	{
		get_char( i, (char_data*)&raw_input, &label);
		for(int i=0; i<IMG_H; i++) {
		for(int j=0; j<IMG_W; j++) {
			int32_t tmp = raw_input[0][i*IMG_W+j]/2;
			input[0][0][i][j] = tmp > 127? 127 : tmp;
		}
		}

		/* Run the neural network inference */
		entry(input, result);
		int r = index_of_max( result );

		if( label == r ) {
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

