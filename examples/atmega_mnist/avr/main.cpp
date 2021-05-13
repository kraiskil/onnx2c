/* Example program that to demonstrate running neural network
 * inference recognizing hand-written numbers.
 * This is intended to run on an Arduino Uno with a
 * Velleman WM412 touch-display shield.
 * See the accompanying Makefile in this and parent directory
 * for more details.
 */


#include "pinout.h"
#include "Adafruit_GFX.h"
#include "MCUFRIEND_kbv.h"
#include "utility/mcufriend_shield.h"
#include "TouchScreen.h"

// Z value limit for registering a touch on the touch screen
#define TOUCH_LIMIT 200
// X and Y values reported at the corners of the touch screen.
// These values must be calibrated, maybe for every screen separately
const int MIN_Y = 133;
const int MAX_Y = 912;
const int MIN_X = 91;
const int MAX_X = 903;

TouchScreen ts = TouchScreen(XP, YP, XM, YM, XP_XM_resistance);
MCUFRIEND_kbv tft;

// older versions of avr-libc lack these?
#ifndef INT32_MAX
#define INT32_MAX 0x7fffffff
#define INT32_MIN 0x80000000
#endif
// This is from the onnx2c-generated C source.
extern "C" {
void entry(int8_t tensor_network_input[1][1][14][14], int8_t tensor_network_output[1][10]);
}

// Input and output buffers to/from the neural network
#define IMG_W 14
#define IMG_H 14
// Indices are: batch, channel, y, x
int8_t input[1][1][IMG_H][IMG_W];
int8_t result[1][10];



/* The TouchScreen and TFT share some pins (A2, A3).
 * After getting a touch event, drawing doesn't work
 * until pins are reset. Touch library does reset pins,
 * but the display library doesn't.
 */
static inline void pins_to_lcd(void)
{
	pinMode(A2, OUTPUT);
	pinMode(A3, OUTPUT);
}

#define BLACK   0x0000
#define BLUE    0x001F
#define CYAN    0x07FF
#define WHITE   0xFFFF

// pixel coordinates for the button rows and columns
// offsets from the star coordinate (X_0, Y_0)
#define ROW_0 0
#define ROW_1 50
#define ROW_2 100
#define ROW_3 150
#define COL_0 0
#define COL_1 50
#define COL_2 100
#define X_0   40
#define Y_0   50
// Button size
#define B_W   40
#define B_H   40

// input draw area to NN input oversampling ratio
#define DISPLAY_OS 8
#define INPUT_START_X 180
#define INPUT_START_Y 60
#define INPUT_W (IMG_W * DISPLAY_OS)
#define INPUT_H (IMG_H * DISPLAY_OS)
/* Draw the rectangle that is the drawing area for user input.
 * This also "clears input", since the drawed data is kept in display memory.
 */
static inline void draw_draw_area(void)
{
	tft.fillRect(INPUT_START_X, INPUT_START_Y, INPUT_W, INPUT_H, WHITE);
	tft.drawRect(INPUT_START_X-1, INPUT_START_Y-1, INPUT_W+2, INPUT_H+2, CYAN);
}


void draw_button(int no, bool selected)
{
	Adafruit_GFX_Button button;
	int x = X_0;
	int y = Y_0;

	switch( no ) {
	case 1:
	case 4:
	case 7:
		x += COL_0;
		break;
	case 0:
	case 2:
	case 5:
	case 8:
		x += COL_1;
		break;
	case 3:
	case 6:
	case 9:
		x += COL_2;
		break;
	}
	switch( no ) {
	case 1:
	case 2:
	case 3:
		y += ROW_0;
		break;
	case 4:
	case 5:
	case 6:
		y += ROW_1;
		break;
	case 7:
	case 8:
	case 9:
		y += ROW_2;
		break;
	case 0:
		y += ROW_3;
		break;
	}

	char txt[] = {'0', 0};
	txt[0]+=no;
	button.initButton(&tft, x, y,  B_W, B_H, CYAN, WHITE, BLUE, txt, 2);
	button.drawButton(selected); // if passed true, button is drawed with colours inverted
}


static inline bool is_in_draw( int x, int y )
{
	if( x < INPUT_START_X ) return false;
	if( x > (INPUT_START_X+INPUT_W) ) return false;
	if( y < INPUT_START_Y ) return false;
	if( y > (INPUT_START_Y+INPUT_H) ) return false;

	return true;
}

// Debugging utility prints. Not used in the demo.
void print_input(void)
{
	printf("int8_t input[1][1][14][14] = {{{\n");
	for( int i=0; i<IMG_W; i++ ) {
		printf("{");
		for( int j=0; j<IMG_W; j++ )
			printf("%3d, ", input[0][0][i][j]);
		printf("},\n");
	}
	printf("}}};\n");
}
void print_result(void)
{
	printf("int8_t result[1][10] = {{\n");
	for( int i=0; i<10; i++ )
		printf("%3d, ", result[0][i]);
	printf("}};\n");
}


void run_inference(void)
{
	// Read the input area from the display, resample down to the neural
	// network input size
	for( int i=0; i<IMG_H; i++ )
	for( int j=0; j<IMG_W; j++ ) {
		input[0][0][j][i]=0;
		uint16_t colour[DISPLAY_OS*DISPLAY_OS];
		uint16_t sum=0;
		tft.readGRAM(INPUT_START_X + i*DISPLAY_OS, INPUT_START_Y + j*DISPLAY_OS, colour, DISPLAY_OS, DISPLAY_OS);
		for( int k=0; k<(DISPLAY_OS*DISPLAY_OS); k++)
		{
			sum += (colour[k] < 10 ? 32 : 0);
		}
		input[0][0][j][i] = sum > 127 ? 127 : sum;
	}

	// Run neural network inference. Pulse pin PC5 for measureing timing.
	PORTC |= 1<<5;
	entry(input, result);
	PORTC &= ~(1<<5);

	// Re-draw the buttons, highlight the button with the result
	int8_t max=-127;
	for( int i=0; i<10; i++ )
		max = result[0][i] > max ? result[0][i] : max;

	for( int i=0; i<10; i++ ) {
		if( result[0][i] == max && max > 0)
			draw_button(i, true);
		else
			draw_button(i, false);
	}
}

// Track the loop times with no press events
// Kind of a debounce for when the touch screen misses an input
// in the middle of the user drawing
#define NOPRESS_LIMIT 500
static int nopress_ctr=0;

void handle_touch_event( int pixel_x, int pixel_y )
{
	// "draw" the user input,
	if( is_in_draw( pixel_x, pixel_y ) ) {
		tft.fillRect(pixel_x-3, pixel_y-3, 7, 7, BLACK);
	}
	// "magic" area in the upper left hand corner
	else if( pixel_x < 50 && pixel_y < 50 ) {
		//print_input();
		//print_result();
		draw_draw_area();
	}
	// anywhere else on the left hand side of the screen
	else if( pixel_x < (COL_2+B_W) ) {
		draw_draw_area();
	}
}

// Yes, this code originated as a Arduino IDE scetch :)
static inline void loop_(void)
{
	TSPoint p;
	p = ts.getPoint();
	// NB: x and y swapped, since screen is rotated to landscape
	int pixel_x, pixel_y;
	pixel_x = map(p.y, MIN_X, MAX_X, 0, tft.width());
	pixel_y = map(p.x, MIN_Y, MAX_Y, 0, tft.height());

	pins_to_lcd();

	if( p.z < TOUCH_LIMIT ){
		nopress_ctr++;
		if (nopress_ctr > NOPRESS_LIMIT){
			run_inference();
			nopress_ctr=0;
		}
	}
	else {
		nopress_ctr=0;
		handle_touch_event(pixel_x, pixel_y);
	}
}

int main(void)
{
	// Initialize the Arduino framework.
	// init() e.g. sets up timers so delay() works (which is used
	// in the libraries written for Arduino - that we use here)
	// init() disables host UART - revert that disable here without
	// needing to enable the Serial module. Using stdio/printf saves
	// 8% of RAM :)
	int16_t tmp = UCSR0B;
	init();
	UCSR0B = tmp;

	// Use PC5 (labeled SDA) for toggling a GPIO for timing measurement
	DDRC |= 1<<5;

	// Save maybe a few bytes here by hard-coding the ID
	//uint16_t ID = tft.readID();
	tft.begin(/*ID=*/0x9341);
	tft.setRotation(1);

	tft.fillScreen(BLACK);

	for( int i=0; i<10; i++)
		draw_button(i, false);
	draw_draw_area();

	while(1){
		loop_();
	}
	return 0;
}

