#!/bin/bash
#
# Run benchmarks that fit on STM32F411.
# Print a comparison to values from a previos
# run. These values are (for now) hard-coded into
# the script

print_results()
{
	b_rom=$1
	rom=$4
	rom_c=$(python -c "print($rom.0/$b_rom*100)")
	b_ram=$2
	ram=$5
	ram_c=$(python -c "print($ram.0/$b_ram*100)")
	b_time=$3
	time=$6
	time_c=$(python -c "print($time.0/$b_time*100)")
echo "-     was       now         relatve-%
     rom:  $rom       $b_rom      $rom_c
     ram:  $ram       $b_ram      $ram_c
     time: $time      $b_time     $time_c" | column -t -s ' '
}

echo  -e "\n Running benchmark_conv_fits_128k"
conv_fits_baseline="37712 120412 333"
conv_fits=$($1/scripts/measure_stm32f411_nucleo.sh $1/test/benchmarks/benchmark_conv_fits_128k/model.onnx |tail -n 3)
print_results $conv_fits_baseline $conv_fits

