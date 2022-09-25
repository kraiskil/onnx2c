#!/bin/bash
# (does that work on a mac? :) )
#
# Helper script to run the onnx2c benchmarks.
# The benchmark is built into the binary `onnx2c_benchmark`,
# that uses the Google benchmark library and a bunch (currently too few)
# samples of real-world nodes. This means it runs single operands, such
# as Conv2D, with real-world input sizes.
#
# This script should be run as part of 'make run_benchmark'.
#
# Since it benchmarking depends heavily on the computer it runs on,
# the comparison is done running against the benchmark run on the master
# branch of onnx2c. This means to be able to run this benchmark, you must
# first compile the master version and run this benchmark. It will create
# the result file 'benchmark_result_master.json'. When running this script
# when on a onnx2c branch, the script will compare the branch results agaist
# the master results.
#
# NB: the testing is not thorough, since it runs a sample of 1. This script
# uses the `compare.py` script from Google Benchmark. Pleas see the note
# on U tests at https://github.com/google/benchmark/blob/main/docs/tools.md

THIS_SCRIPT=$0
SCRIPT_LOCATION=$(dirname $THIS_SCRIPT)
ONNX2C_REPO=$(cd $SCRIPT_LOCATION; git rev-parse --show-toplevel)
ONNX2C_BRANCH=$(cd $ONNX2C_REPO; git rev-parse --abbrev-ref HEAD)

ONNX2C_BENCHMARK_BIN=onnx2c_benchmark

echo current dir is $PWD
./$ONNX2C_BENCHMARK_BIN --benchmark_out_format=json --benchmark_out=$ONNX2C_BRANCH.json

if [[ $ONNX2C_BRANCH == master ]]
then
	exit 0
fi
# "else" run compare against master.json
if [[ ! -e master.json ]]
then
	echo  ERROR: no master.json file found. Benchmark has not been run from master branch!
	echo  Nothing to compare against.
	exit 1
fi

python3 $ONNX2C_REPO/benchmark/tools/compare.py benchmarks master.json $ONNX2C_BRANCH.json

