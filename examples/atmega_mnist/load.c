/* Utility functions to load the MNIST dataset.
 * The dataset was downloaded when running pytorch.
 * If not, see the paths below where to put it
 */
#include <stdio.h>
#include "load.h"

// The PyTorch script downloads the MNIST dataset into this relative path
#define DATA_FILE_PATH  "data/MNIST/raw/t10k-images-idx3-ubyte"
#define LABEL_FILE_PATH "data/MNIST/raw/t10k-labels-idx1-ubyte"

FILE *fd_data = NULL;
FILE *fd_label = NULL;

void open_files(void)
{
	fd_data = fopen(DATA_FILE_PATH, "rb");
	if( fd_data == NULL )
	{
		fprintf(stderr, "failure opening data file\n");
	}

	fd_label = fopen(LABEL_FILE_PATH, "rb");
	if( fd_data == NULL )
	{
		fprintf(stderr, "failure opening label file\n");
	}
}

void get_char( int index, char_data *d, int *label)
{
	uint8_t l;
	uint8_t data[28][28];

	fseek(fd_label, 8 + index, SEEK_SET);
	int r = fread(&l, 1, 1, fd_label);
	*label=l;
	(void) r;

	fseek(fd_data, 16 + index*28*28, SEEK_SET);
	r = fread(data, 28*28, 1, fd_data);
	(void) r;

	// TODO: is this downsampling ok?
	for( int r=0; r<IMG_H; r++ )
	for( int c=0; c<IMG_W; c++ )
	{
		int32_t sum=0;
		sum += data[r*2][c*2];
		sum += data[r*2][c*2+1];
		sum += data[r*2+1][c*2];
		sum += data[r*2+1][c*2+1];
		(*d)[r][c] = sum/4;
	}
}

void get_char_float( int index, char_float *f, int *label)
{
	char_data d;
	get_char(index, &d, label);
	for( int r=0; r<IMG_H; r++ )
	for( int c=0; c<IMG_W; c++ )
		(*f)[r][c] = (float)(d[r][c])/255.0;
}

