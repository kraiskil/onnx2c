// Pinouts for Velleman VM412
// Some pins are shared between touch and LCD
#include "Arduino.h"

// Touch device interface
#define YP A3
#define XM A2
#define YM 9
#define XP 8
// Measured resistance between XP and XM, in ohm.
// Seems this is a design specific, or maybe board specific value.
#define XP_XM_resistance 360


// LCD/TFT display
#define LCD_RD   A0
#define LCD_WR   A1
#define LCD_RS   A2
#define LCD_CS   A3
#define LCD_REST A4


