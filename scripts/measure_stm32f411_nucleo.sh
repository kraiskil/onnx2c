#!/bin/bash -e

#
# This script takes a onnx graph file
# and compiles it to run on a STM32F411 NUCLEO
# development board (with an ARM Cortex-M4 MCU
# on it). The execution memory usage and execution
# time is reported.

THIS_SCRIPT_DIR=$(dirname $0)
ONNX_FILE=$1
SERIAL_PORT=/dev/ttyACM0

GENERATED_C=$(basename $ONNX_FILE .onnx).c
GENERATED_ELF=$(basename $ONNX_FILE .onnx).elf

OPENOCD=openocd
OPENOCD_CFG=$THIS_SCRIPT_DIR/openocd.cfg


#Compiler settings. TODO: if these can be taken from
# a separate file, this script could be generic?
CROSS_CC=arm-none-eabi-gcc
CROSS_CFLAGS="-Wall "
CROSS_CFLAGS+="-O4 "
CROSS_CFLAGS+="-mthumb -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 "
CROSS_CFLAGS+="-DSTM32F4 -I${OPENCM3_DIR}/include "

CROSS_LDFLAGS="-nostartfiles "
CROSS_LDFLAGS+="-L${OPENCM3_DIR}/lib "
CROSS_LDFLAGS+="-T${THIS_SCRIPT_DIR}/nucleo-f411re.ld "
CROSS_LDFLAGS+="-Wl,--print-memory-usage "
CROSS_LIBS="-lopencm3_stm32f4 -lc -lgcc -lnosys "

print_usage()
{
	echo  Usage: $0 graph_file.onnx
	echo
	exit 1
}

check_is_in_path()
{
	if ! command -v $1 &> /dev/null
	then
		echo "$1 could not be found in PATH"
		exit 1
	fi
}

check_is_set()
{
	if [[ -z $(printenv $1) ]]
	then
		echo variable $1 is not set
		exit 1
	fi
}

check_tools()
{
	check_is_in_path onnx2c
	check_is_in_path $CROSS_CC
	check_is_in_path $OPENOCD
	check_is_set     OPENCM3_DIR
}


generate_c()
{

	echo generating c from $ONNX_FILE
	# Compile the onnx file to C source
	onnx2c $ONNX_FILE > $GENERATED_C

	# Parse the generated C source, and generate a wrapper file that calls the
	# neural network inference once.
	# 'entry' is the inference function name.
	# Create buffers for the network input and output
	grep entry $GENERATED_C | tr '{' ';' > wrapper.c
	grep entry $GENERATED_C | cut -f 2 -d '(' | cut -d ')' -f 1 | tr ',' ';' >> wrapper.c
	echo  ";" >> wrapper.c
	# Create the function that main() calls
	echo "void run_benchmark(void) {" >> wrapper.c
	echo "entry(" >> wrapper.c
	grep entry $GENERATED_C | cut -f 2 -d '(' | cut -d ')' -f 1 |  sed -e 's/\[[0-9]\+\]//g' | sed -e 's/[a-z_0-9]*\s//g' >> wrapper.c
	echo ");" >> wrapper.c
	echo } >> wrapper.c
}

compile()
{
	echo compiling

	# Baseline. This .elf is ignored after printout of memory sizes
	echo Memory usage WITHOUT the neural network:
	SOURCES="$THIS_SCRIPT_DIR/target_main.c"
	$CROSS_CC $CROSS_CFLAGS $CROSS_LDFLAGS $SOURCES -o $GENERATED_ELF $CROSS_LIBS -DDONT_RUN_TARGET

	# Compile the full benchmark
	echo Memory usage WITH the neural network:
	SOURCES="$THIS_SCRIPT_DIR/target_main.c wrapper.c $GENERATED_C"
	$CROSS_CC $CROSS_CFLAGS $CROSS_LDFLAGS $SOURCES -o $GENERATED_ELF $CROSS_LIBS
}

flash_and_read()
{
	echo Halting the target
	# And clear any old output from the serial port
	$OPENOCD -f $OPENOCD_CFG -c "init; reset halt; exit" -l openocd_log.txt 2> /dev/null
	stty -F $SERIAL_PORT speed 9600 > /dev/null
	timeout 1s cat $SERIAL_PORT > /dev/null || true

	echo flashing
	$OPENOCD -f $OPENOCD_CFG -c "program $GENERATED_ELF verify reset exit" -l openocd_log.txt 2> /dev/null
	timeout 2s cat $SERIAL_PORT > exec_times.txt || true
	cat exec_times.txt
}

if [[ $# != 1 ]]
then
	echo $#
	print_usage
fi

check_tools

generate_c

compile

flash_and_read

vals=$(arm-none-eabi-size  $GENERATED_ELF | tail -n 1)
rom_size=$(echo $vals | cut -d ' ' -f1)
data_size=$(echo $vals | cut -d ' ' -f2)
bss_size=$(echo $vals | cut -d ' ' -f3)
exec_time=$(grep "^Execution time: [0-9]\+ ms$" exec_times.txt  |head -n1 |cut -d ' ' -f 3)

# Print out the raw data so other scripts can get it with a "|tail -n3"
echo "In summary - mostly for scripts"
echo "ROM/Flash usage, heap (data+bss) usage, and runtime in ms:"
echo $rom_size
echo $(($data_size + $bss_size))
echo $exec_time
