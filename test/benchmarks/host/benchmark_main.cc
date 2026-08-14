#include <benchmark/benchmark.h>
// Work-around the generated C files including "math.h", and causing
//  error: '::acos' has not been declared
#include <cmath>
#include <sys/resource.h>

namespace benchmark_meta {

struct ModelMeta {
	int64_t im2col_bytes;
	bool heuristic_selected;
};

constexpr ModelMeta yolov6n_biggestconv = {
    1LL * 160 * 160 * 32 * 3 * 3 * sizeof(float),
    true,
};

constexpr ModelMeta yolov6n_inputlayer = {
    1LL * 320 * 320 * 3 * 3 * 3 * sizeof(float),
    true,
};

constexpr ModelMeta yolov6n_lastconv = {
    1LL * 20 * 20 * 128 * 1 * 1 * sizeof(float),
    false,
};

constexpr ModelMeta conv_fits_128k = {
    1LL * 20 * 20 * 28 * 3 * 3 * sizeof(float),
    true,
};

constexpr ModelMeta resnet_stem_7x7 = {
    1LL * 112 * 112 * 3 * 7 * 7 * sizeof(float),
    true,
};

constexpr ModelMeta resnet_3x3 = {
    1LL * 56 * 56 * 64 * 3 * 3 * sizeof(float),
    true,
};

constexpr ModelMeta resnet_bottleneck_1x1 = {
    1LL * 28 * 28 * 256 * 1 * 1 * sizeof(float),
    true,
};

constexpr ModelMeta mobilenet_depthwise_3x3 = {
    1LL * 112 * 112 * 1 * 3 * 3 * sizeof(float),
    true,
};

constexpr ModelMeta mobilenet_pointwise_1x1 = {
    1LL * 112 * 112 * 32 * 1 * 1 * sizeof(float),
    false,
};

static int64_t peak_rss_bytes()
{
	struct rusage usage;
	if (getrusage(RUSAGE_SELF, &usage) != 0)
		return 0;
#if defined(__APPLE__)
	return usage.ru_maxrss;
#else
	return usage.ru_maxrss * 1024LL;
#endif
}

static void set_counters(benchmark::State& state, const ModelMeta& meta, const char* policy)
{
	bool selected = false;
	if (policy[0] == 'a')
		selected = true;
	else if (policy[0] == 'h')
		selected = meta.heuristic_selected;

	state.counters["im2col_extra_bytes_theoretical"] = selected ? meta.im2col_bytes : 0;
	state.counters["peak_memory_bytes_measured"] = peak_rss_bytes();
	state.counters["num_convs"] = 1;
	state.counters["num_im2col_convs"] = selected ? 1 : 0;
	state.counters["im2col_fraction"] = selected ? 1.0 : 0.0;
}

} // namespace benchmark_meta

#define DEFINE_BENCH(SYMBOL, POLICY, META)                                          \
	static void BM_##SYMBOL##_##POLICY(benchmark::State& state)                 \
	{                                                                           \
		for (auto _ : state) {                                              \
			entry(X, W, Y);                                             \
		}                                                                   \
		benchmark_meta::set_counters(state, benchmark_meta::META, #POLICY); \
	}                                                                           \
	BENCHMARK(BM_##SYMBOL##_##POLICY)->Name(#SYMBOL "/" #POLICY)

namespace conv_yolov6n_biggestconv_none {
#include "conv_yolov6n_biggestconv_none.c"
float X[1][32][160][160];
float W[32][32][3][3];
float Y[1][32][160][160];
DEFINE_BENCH(conv_yolov6n_biggestconv, none, yolov6n_biggestconv);
} // namespace conv_yolov6n_biggestconv_none

namespace conv_yolov6n_biggestconv_all {
#include "conv_yolov6n_biggestconv_all.c"
float X[1][32][160][160];
float W[32][32][3][3];
float Y[1][32][160][160];
DEFINE_BENCH(conv_yolov6n_biggestconv, all, yolov6n_biggestconv);
} // namespace conv_yolov6n_biggestconv_all

namespace conv_yolov6n_biggestconv_heuristic {
#include "conv_yolov6n_biggestconv_heuristic.c"
float X[1][32][160][160];
float W[32][32][3][3];
float Y[1][32][160][160];
DEFINE_BENCH(conv_yolov6n_biggestconv, heuristic, yolov6n_biggestconv);
} // namespace conv_yolov6n_biggestconv_heuristic

namespace conv_yolov6n_inputlayer_none {
#include "conv_yolov6n_inputlayer_none.c"
float X[1][3][640][640];
float W[16][3][3][3];
float Y[1][16][320][320];
DEFINE_BENCH(conv_yolov6n_inputlayer, none, yolov6n_inputlayer);
} // namespace conv_yolov6n_inputlayer_none

namespace conv_yolov6n_inputlayer_all {
#include "conv_yolov6n_inputlayer_all.c"
float X[1][3][640][640];
float W[16][3][3][3];
float Y[1][16][320][320];
DEFINE_BENCH(conv_yolov6n_inputlayer, all, yolov6n_inputlayer);
} // namespace conv_yolov6n_inputlayer_all

namespace conv_yolov6n_inputlayer_heuristic {
#include "conv_yolov6n_inputlayer_heuristic.c"
float X[1][3][640][640];
float W[16][3][3][3];
float Y[1][16][320][320];
DEFINE_BENCH(conv_yolov6n_inputlayer, heuristic, yolov6n_inputlayer);
} // namespace conv_yolov6n_inputlayer_heuristic

namespace conv_yolov6n_lastconv_none {
#include "conv_yolov6n_lastconv_none.c"
float X[1][128][20][20];
float W[1][128][1][1];
float Y[1][1][20][20];
DEFINE_BENCH(conv_yolov6n_lastconv, none, yolov6n_lastconv);
} // namespace conv_yolov6n_lastconv_none

namespace conv_yolov6n_lastconv_all {
#include "conv_yolov6n_lastconv_all.c"
float X[1][128][20][20];
float W[1][128][1][1];
float Y[1][1][20][20];
DEFINE_BENCH(conv_yolov6n_lastconv, all, yolov6n_lastconv);
} // namespace conv_yolov6n_lastconv_all

namespace conv_yolov6n_lastconv_heuristic {
#include "conv_yolov6n_lastconv_heuristic.c"
float X[1][128][20][20];
float W[1][128][1][1];
float Y[1][1][20][20];
DEFINE_BENCH(conv_yolov6n_lastconv, heuristic, yolov6n_lastconv);
} // namespace conv_yolov6n_lastconv_heuristic

namespace conv_fits_128k_none {
#include "conv_fits_128k_none.c"
float X[1][28][20][20];
float W[28][28][3][3];
float Y[1][28][20][20];
DEFINE_BENCH(conv_fits_128k, none, conv_fits_128k);
} // namespace conv_fits_128k_none

namespace conv_fits_128k_all {
#include "conv_fits_128k_all.c"
float X[1][28][20][20];
float W[28][28][3][3];
float Y[1][28][20][20];
DEFINE_BENCH(conv_fits_128k, all, conv_fits_128k);
} // namespace conv_fits_128k_all

namespace conv_fits_128k_heuristic {
#include "conv_fits_128k_heuristic.c"
float X[1][28][20][20];
float W[28][28][3][3];
float Y[1][28][20][20];
DEFINE_BENCH(conv_fits_128k, heuristic, conv_fits_128k);
} // namespace conv_fits_128k_heuristic

namespace conv_resnet_stem_7x7_none {
#include "conv_resnet_stem_7x7_none.c"
float X[1][3][224][224];
float W[64][3][7][7];
float Y[1][64][112][112];
DEFINE_BENCH(conv_resnet_stem_7x7, none, resnet_stem_7x7);
} // namespace conv_resnet_stem_7x7_none

namespace conv_resnet_stem_7x7_all {
#include "conv_resnet_stem_7x7_all.c"
float X[1][3][224][224];
float W[64][3][7][7];
float Y[1][64][112][112];
DEFINE_BENCH(conv_resnet_stem_7x7, all, resnet_stem_7x7);
} // namespace conv_resnet_stem_7x7_all

namespace conv_resnet_stem_7x7_heuristic {
#include "conv_resnet_stem_7x7_heuristic.c"
float X[1][3][224][224];
float W[64][3][7][7];
float Y[1][64][112][112];
DEFINE_BENCH(conv_resnet_stem_7x7, heuristic, resnet_stem_7x7);
} // namespace conv_resnet_stem_7x7_heuristic

namespace conv_resnet_3x3_none {
#include "conv_resnet_3x3_none.c"
float X[1][64][56][56];
float W[64][64][3][3];
float Y[1][64][56][56];
DEFINE_BENCH(conv_resnet_3x3, none, resnet_3x3);
} // namespace conv_resnet_3x3_none

namespace conv_resnet_3x3_all {
#include "conv_resnet_3x3_all.c"
float X[1][64][56][56];
float W[64][64][3][3];
float Y[1][64][56][56];
DEFINE_BENCH(conv_resnet_3x3, all, resnet_3x3);
} // namespace conv_resnet_3x3_all

namespace conv_resnet_3x3_heuristic {
#include "conv_resnet_3x3_heuristic.c"
float X[1][64][56][56];
float W[64][64][3][3];
float Y[1][64][56][56];
DEFINE_BENCH(conv_resnet_3x3, heuristic, resnet_3x3);
} // namespace conv_resnet_3x3_heuristic

namespace conv_resnet_bottleneck_1x1_none {
#include "conv_resnet_bottleneck_1x1_none.c"
float X[1][256][28][28];
float W[64][256][1][1];
float Y[1][64][28][28];
DEFINE_BENCH(conv_resnet_bottleneck_1x1, none, resnet_bottleneck_1x1);
} // namespace conv_resnet_bottleneck_1x1_none

namespace conv_resnet_bottleneck_1x1_all {
#include "conv_resnet_bottleneck_1x1_all.c"
float X[1][256][28][28];
float W[64][256][1][1];
float Y[1][64][28][28];
DEFINE_BENCH(conv_resnet_bottleneck_1x1, all, resnet_bottleneck_1x1);
} // namespace conv_resnet_bottleneck_1x1_all

namespace conv_resnet_bottleneck_1x1_heuristic {
#include "conv_resnet_bottleneck_1x1_heuristic.c"
float X[1][256][28][28];
float W[64][256][1][1];
float Y[1][64][28][28];
DEFINE_BENCH(conv_resnet_bottleneck_1x1, heuristic, resnet_bottleneck_1x1);
} // namespace conv_resnet_bottleneck_1x1_heuristic

namespace conv_mobilenet_depthwise_3x3_none {
#include "conv_mobilenet_depthwise_3x3_none.c"
float X[1][32][112][112];
float W[32][1][3][3];
float Y[1][32][112][112];
DEFINE_BENCH(conv_mobilenet_depthwise_3x3, none, mobilenet_depthwise_3x3);
} // namespace conv_mobilenet_depthwise_3x3_none

namespace conv_mobilenet_depthwise_3x3_all {
#include "conv_mobilenet_depthwise_3x3_all.c"
float X[1][32][112][112];
float W[32][1][3][3];
float Y[1][32][112][112];
DEFINE_BENCH(conv_mobilenet_depthwise_3x3, all, mobilenet_depthwise_3x3);
} // namespace conv_mobilenet_depthwise_3x3_all

namespace conv_mobilenet_depthwise_3x3_heuristic {
#include "conv_mobilenet_depthwise_3x3_heuristic.c"
float X[1][32][112][112];
float W[32][1][3][3];
float Y[1][32][112][112];
DEFINE_BENCH(conv_mobilenet_depthwise_3x3, heuristic, mobilenet_depthwise_3x3);
} // namespace conv_mobilenet_depthwise_3x3_heuristic

namespace conv_mobilenet_pointwise_1x1_none {
#include "conv_mobilenet_pointwise_1x1_none.c"
float X[1][32][112][112];
float W[64][32][1][1];
float Y[1][64][112][112];
DEFINE_BENCH(conv_mobilenet_pointwise_1x1, none, mobilenet_pointwise_1x1);
} // namespace conv_mobilenet_pointwise_1x1_none

namespace conv_mobilenet_pointwise_1x1_all {
#include "conv_mobilenet_pointwise_1x1_all.c"
float X[1][32][112][112];
float W[64][32][1][1];
float Y[1][64][112][112];
DEFINE_BENCH(conv_mobilenet_pointwise_1x1, all, mobilenet_pointwise_1x1);
} // namespace conv_mobilenet_pointwise_1x1_all

namespace conv_mobilenet_pointwise_1x1_heuristic {
#include "conv_mobilenet_pointwise_1x1_heuristic.c"
float X[1][32][112][112];
float W[64][32][1][1];
float Y[1][64][112][112];
DEFINE_BENCH(conv_mobilenet_pointwise_1x1, heuristic, mobilenet_pointwise_1x1);
} // namespace conv_mobilenet_pointwise_1x1_heuristic

BENCHMARK_MAIN();
