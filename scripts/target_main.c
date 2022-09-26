/* This file is part of onnx2c.
 *
 * Generic runner script for onnx2c generated
 * neural networks. This is linked against
 * LibOpenCM3 to be run on a STM32F411 Nucleo board.
 * It prints out execution time on the board's
 * USB-UART.
 */

#include <stdio.h>
#include <libopencm3/cm3/systick.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/usart.h>

void run_benchmark(void);

void system_setup(void)
{
	const struct rcc_clock_scale *clocks = &rcc_hsi_configs[RCC_CLOCK_3V3_84MHZ];
	rcc_clock_setup_pll( clocks );

	rcc_periph_clock_enable(RCC_USART2);
	rcc_periph_clock_enable(RCC_GPIOA);

	gpio_mode_setup(GPIOA, GPIO_MODE_AF, GPIO_PUPD_NONE, GPIO2 | GPIO3);
	gpio_set_af(GPIOA, GPIO_AF7, GPIO2 | GPIO3);

	usart_set_baudrate(USART2, 9600);
	usart_set_databits(USART2, 8);
	usart_set_stopbits(USART2, USART_STOPBITS_1);
	usart_set_mode(USART2, USART_MODE_TX);
	usart_set_parity(USART2, USART_PARITY_NONE);
	usart_set_flow_control(USART2, USART_FLOWCONTROL_NONE);

	usart_enable(USART2);

	/* Enable systick at a 1ms interrupt rate */
	systick_set_reload(84000);
	systick_set_clocksource(STK_CSR_CLKSOURCE_AHB);
	systick_counter_enable();
	systick_interrupt_enable();
}

// Systick timer set to 1ms
static volatile uint32_t system_millis=0;
void sys_tick_handler(void)
{
	system_millis++;
}
// Callback for printf() to push the characters to the UART
int _write(int fd, char *ptr, int len)
{
	int i = 0;

	while (*ptr && (i < len)) {
		usart_send_blocking(USART2, *ptr);
		i++;
		ptr++;
	}
	return i;
}


int main(void)
{
	system_setup();

	for( int i=0; i<10; i++) {
		uint32_t tick = system_millis;
		// Just to print out the baseline memory usage - printf is big :)
		#ifndef DONT_RUN_TARGET
		run_benchmark();
		#endif
		printf("Execution time: %ldms\n", system_millis-tick);
	}

	return 0;
}

