#include <libopencm3/stm32/gpio.h>
#include <libopencm3/stm32/rcc.h>
#include <libopencm3/stm32/timer.h>
#include <math.h>

void entry(float x[1][1], float sin_x[1][1]);


static void periph_setup(void)
{
	rcc_clock_setup_pll(&rcc_hse_8mhz_3v3[RCC_CLOCK_3V3_84MHZ]);
	rcc_periph_clock_enable(RCC_GPIOD);

	/* PD13 for measuring execution time */
	gpio_mode_setup(GPIOD, GPIO_MODE_OUTPUT, GPIO_PUPD_NONE, GPIO13);

	/* PD12 is connected to TIM4, channel 1 - this is the PWM LED*/
	gpio_mode_setup(GPIOD, GPIO_MODE_AF,
			GPIO_PUPD_NONE, GPIO12);
	gpio_set_af(GPIOD, GPIO_AF2, GPIO12);
	rcc_periph_clock_enable(RCC_TIM4);
	timer_set_mode(TIM4, TIM_CR1_CKD_CK_INT, TIM_CR1_CMS_CENTER_1, TIM_CR1_DIR_UP);
	timer_set_oc_mode(TIM4, TIM_OC1, TIM_OCM_PWM1);
	timer_enable_oc_output(TIM4, TIM_OC1);

	/* Set timer to run full speed, counting to 200.
	 * The neural network will then modify the comparator
	 * value to set PWM duty cycle*/
	timer_set_prescaler(TIM4, 1);
	timer_continuous_mode(TIM4);
	timer_set_period(TIM4, 200);
	timer_set_oc_value(TIM4, TIM_OC1, 200);
	timer_enable_counter(TIM4);
}

int main(void)
{
	periph_setup();

	float x[1][1] = {{0}};
	float rv[1][1] = {{0}};


	while (1) {

		// Loop input through the range used to
		// train the model (i.e. 0-2pi). Outside of
		// that range, the model just gets confused
		x[0][0] += 0.1;
		if( x[0][0] > 2 * M_PI)
			x[0][0] = 0;

		// run inference, and toggle GPIO to allow timing
		// measurements
		gpio_set(GPIOD, GPIO13);
		entry(x, rv);
		gpio_clear(GPIOD, GPIO13);

		// shift output to be a sine wave with range
		// 0-200, which is also the bounds of the PWM
		// duty cycle
		float y = rv[0][0];
		y = (y+1) * 100;
		if( y>200)
			y=200;
		if( y<0)
			y=0;
		timer_set_oc_value(TIM4, TIM_OC1, (int16_t)y);

		/* Slow down the update of the PWM value for
		 * a nice pulsating effect */
		for(int i=0; i<840000; i++)
			__asm__("nop");
	}

	return 0;
}
