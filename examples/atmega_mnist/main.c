#include <stdio.h>
#include <util/delay.h>
#include <avr/io.h>

#include "sample_7.h"
//int8_t input[1][1][28][28];

void entry(int8_t input[1][1][28][28], int8_t output[1][10]);

int8_t find_max(int8_t result[1][10]);
void show_result(int8_t res);

int main(void)
{
	DDRD = _BV(2)|_BV(3)|_BV(4);
	DDRB = _BV(5);

	int8_t result[1][10]={{0}};
	while(1) {
		PORTB |= _BV(5);
		entry(input, result);
		PORTB &= ~_BV(5);

		_delay_ms(1000);
		printf("result:\n");
		for(int i=0;i<10;i++)
			printf("%d: %d\n", i, result[0][i]);

		int8_t res = find_max(result);
		printf("max res %d\n", res);
		show_result(res);
	}
	return 0;
}


int8_t find_max(int8_t result[1][10])
{
	int8_t max=-127;
	int8_t res=-1;
	for( int i=0; i<10; i++) {
		if( result[0][i] > max ) {
			res = i;
			max = result[0][i];
		}
	}
	return res;
}

/* Shows 'res' on output, if res is in [0,9],
 * blank if it is outside that range */
void show_result(int8_t res)
{
	// TODO: get a 7-segment display
	if( res == 4 ) // green
		PORTD=  _BV(2);
	else if( res == 7 ) // yellow
		PORTD = _BV(3);
	else
		PORTD = _BV(4);  // red
}

