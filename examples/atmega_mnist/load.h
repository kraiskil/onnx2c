#pragma once
#include <stdint.h>
#define NUM_TESTS 10000
//#define NUM_TESTS 1

#define IMG_W 14
#define IMG_H 14
#define IMAGE_SIZE (IMG_H * IMG_W)
typedef uint8_t char_data[IMG_H][IMG_W];
typedef float char_float[IMG_H][IMG_W];

void open_files(void);
void get_char( int index, char_data *, int *label);
void get_char_float( int index, char_float *, int *label);

