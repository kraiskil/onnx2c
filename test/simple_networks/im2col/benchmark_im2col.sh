#!/bin/bash
# Benchmark script to compare im2col transformation performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="/workspace/build"
MODEL_DIR="$SCRIPT_DIR"
MODEL="multi_conv_bench"

echo "============================================"
echo "  Im2col Benchmark: 2-layer Conv Network"
echo "  (3x16x16 -> 8x16x16 -> 8x16x16)"
echo "============================================"
echo ""

# Generate the benchmark model if it doesn't exist
if [ ! -f "$MODEL_DIR/${MODEL}.onnx" ]; then
    echo "[0/4] Generating benchmark ONNX model..."
    python3 "$MODEL_DIR/generate_benchmark_model.py"
fi

# Generate C code without im2col
echo "[1/4] Generating C code without im2col..."
$BUILD_DIR/onnx2c -l 0 "$MODEL_DIR/${MODEL}.onnx" > "$BUILD_DIR/${MODEL}_no_im2col.c" 2>/dev/null

# Generate C code with im2col
echo "[2/4] Generating C code with im2col..."
$BUILD_DIR/onnx2c -p im2col -l 0 "$MODEL_DIR/${MODEL}.onnx" > "$BUILD_DIR/${MODEL}_with_im2col.c" 2>/dev/null

# Create benchmark wrapper
cat > "$BUILD_DIR/benchmark_wrapper.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void entry(const float* input, float* output);

int main(int argc, char** argv) {
    int iterations = 1000;
    if (argc > 1) iterations = atoi(argv[1]);
    
    // Allocate buffers
    float* input = malloc(3 * 32 * 32 * sizeof(float));
    float* output = malloc(64 * 32 * 32 * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < 3 * 32 * 32; i++) input[i] = 1.0f;
    
    // Warmup
    for (int i = 0; i < 10; i++) entry(input, output);
    
    // Benchmark
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        entry(input, output);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double avg_ms = (elapsed * 1000.0) / iterations;
    
    printf("Iterations: %d\n", iterations);
    printf("Total time: %.3f seconds\n", elapsed);
    printf("Average per iteration: %.3f ms\n", avg_ms);
    printf("Throughput: %.1f iterations/sec\n", iterations / elapsed);
    
    free(input);
    free(output);
    return 0;
}
EOF

# Compile version without im2col
echo "[3/4] Compiling version without im2col..."
gcc -O3 -o "$BUILD_DIR/${MODEL}_no_im2col" "$BUILD_DIR/${MODEL}_no_im2col.c" "$BUILD_DIR/benchmark_wrapper.c" -lm

# Compile version with im2col
echo "      Compiling version with im2col..."
gcc -O3 -o "$BUILD_DIR/${MODEL}_with_im2col" "$BUILD_DIR/${MODEL}_with_im2col.c" "$BUILD_DIR/benchmark_wrapper.c" -lm

# Run benchmarks
echo "[4/4] Running benchmarks..."
echo ""

echo "--------------------------------------------"
echo " WITHOUT im2col:"
echo "--------------------------------------------"
RESULT_NO_IM2COL=$($BUILD_DIR/${MODEL}_no_im2col 1000)
echo "$RESULT_NO_IM2COL"

echo ""
echo "--------------------------------------------"
echo " WITH im2col:"
echo "--------------------------------------------"
RESULT_WITH_IM2COL=$($BUILD_DIR/${MODEL}_with_im2col 1000)
echo "$RESULT_WITH_IM2COL"

# Extract average per iteration time (ms)
AVG_NO_IM2COL=$(echo "$RESULT_NO_IM2COL" | grep "Average per iteration:" | awk '{print $4}')
AVG_WITH_IM2COL=$(echo "$RESULT_WITH_IM2COL" | grep "Average per iteration:" | awk '{print $4}')

# Calculate speedup (inverse - lower time = faster)
SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $AVG_NO_IM2COL / $AVG_WITH_IM2COL}")

echo ""
echo "============================================"
echo "  Average WITHOUT im2col: ${AVG_NO_IM2COL} ms"
echo "  Average WITH im2col:    ${AVG_WITH_IM2COL} ms"
echo "  SPEEDUP: ${SPEEDUP}x faster"
echo "============================================"
