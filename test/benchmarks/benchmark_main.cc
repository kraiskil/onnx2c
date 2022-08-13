#include <benchmark/benchmark.h>
// Work-around the generated C files including "math.h", and causing
//  error: ‘::acos’ has not been declared
#include <cmath>

namespace yolov6n_biggestconv {
//void entry(const float tensor_X[1][32][160][160], const float tensor_w[32][32][3][3], float tensor_Y[1][32][160][160]) {
#include "conv_yolov6n_biggestconv.c"
float X[1][32][160][160];
float W[32][32][3][3];
float Y[1][32][160][160];

static void BM_yolov6n_biggestconv(benchmark::State& state) {

	for (auto _ : state) {
		yolov6n_biggestconv::entry(X, W, Y);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_yolov6n_biggestconv);
}

namespace yolov6n_inputlayer{
#include "conv_yolov6n_inputlayer.c"
float X[1][3][640][640];
float W[16][3][3][3];
float Y[1][16][320][320];
static void BM_yolov6n_inputlayer(benchmark::State& state) {

	for (auto _ : state) {
		entry(X, W, Y);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_yolov6n_inputlayer);
}

namespace yolov6n_lastconv{
#include "conv_yolov6n_lastconv.c"
float X[1][128][20][20];
float W[1][128][1][1];
float Y[1][1][20][20];
static void BM_yolov6n_lastconv(benchmark::State& state) {

	for (auto _ : state) {
		entry(X, W, Y);
	}
}
// Register the function as a benchmark
BENCHMARK(BM_yolov6n_lastconv);
}



// Run the benchmark
BENCHMARK_MAIN();

