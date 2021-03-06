/*
 * Create the input tensor:
 * 5x5 "pixels", with NUM_CHAN channels per pixel
 * Each channel inside a pixel is splatted with the same value.
 * This syntax is a gcc extension. 
 * If this bites you, please write out this with "#ifd NUM_CHAN ==" macros
 * At least cases 1 and 3 channels are used.
 */

float input[1][5][5][NUM_CHAN] = {{
{  {[0 ... NUM_CHAN-1]=1},  {[0 ... NUM_CHAN-1]=2},  {[0 ... NUM_CHAN-1]=3},  {[0 ... NUM_CHAN-1]=4},  {[0 ... NUM_CHAN-1]=5}},
{  {[0 ... NUM_CHAN-1]=6},  {[0 ... NUM_CHAN-1]=7},  {[0 ... NUM_CHAN-1]=8},  {[0 ... NUM_CHAN-1]=9}, {[0 ... NUM_CHAN-1]=10}},
{ {[0 ... NUM_CHAN-1]=11}, {[0 ... NUM_CHAN-1]=12}, {[0 ... NUM_CHAN-1]=13}, {[0 ... NUM_CHAN-1]=14}, {[0 ... NUM_CHAN-1]=15}},
{ {[0 ... NUM_CHAN-1]=16}, {[0 ... NUM_CHAN-1]=17}, {[0 ... NUM_CHAN-1]=18}, {[0 ... NUM_CHAN-1]=19}, {[0 ... NUM_CHAN-1]=20}},
{ {[0 ... NUM_CHAN-1]=21}, {[0 ... NUM_CHAN-1]=22}, {[0 ... NUM_CHAN-1]=23}, {[0 ... NUM_CHAN-1]=24}, {[0 ... NUM_CHAN-1]=25}} }};
