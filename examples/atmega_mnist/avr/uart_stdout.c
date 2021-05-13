/* Implment the stdio backend to use a UART.
 * Code from avr-libc manual.
 * Probably GPL, but the webpage didn't say...
 * By default, sets UART to 9600baud, 8N1 format
 */

#include <stdio.h>
#include <avr/io.h>

static int uart_putchar(char c, FILE *stream);
static FILE mystdout = FDEV_SETUP_STREAM(uart_putchar, NULL,
                                         _FDEV_SETUP_WRITE);
static int
uart_putchar(char c, FILE *stream)
{
	if (c == '\n')
		uart_putchar('\r', stream);
	loop_until_bit_is_set(UCSR0A, UDRE0);
	UDR0 = c;
	return 0;
}

void __attribute__((constructor)) uart_init(void)
{
	UCSR0B |= (1 << RXEN0) | (1 << TXEN0);
	UCSR0C |= (1 << UMSEL01) | (1 << UCSZ00) | (1 << UCSZ01);
	#if F_CPU == 16000000
	UBRR0 = 0x67;  // divisor -> 9600 baud
	#else
	#error set USART rate for your clock freqency
	#endif
	stdout = &mystdout;
}

