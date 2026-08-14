#!/bin/bash
# Benchmark script for ResNet model

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
MODEL="$SCRIPT_DIR/resnet18-v2-7.onnx"
MODEL_NAME="resnet18_bench"

echo "============================================"
echo "  ResNet-18 Benchmark"
echo "  Comparing im2col transformation"
echo "============================================"
echo ""

if [ ! -f "$MODEL" ]; then
    echo "Downloading ResNet-18 model..."
    wget -O "$MODEL" https://huggingface.co/onnxmodelzoo/resnet18-v2-7/resolve/main/resnet18-v2-7.onnx
fi

# Generate C code without im2col (skip if already exists)
echo "[1/4] Generating C code without im2col..."
if [ ! -f "$BUILD_DIR/${MODEL_NAME}_no_im2col.c" ]; then
    $BUILD_DIR/onnx2c -l 0 "$MODEL" > "$BUILD_DIR/${MODEL_NAME}_no_im2col.c" 2>/dev/null
fi

# Generate C code with im2col (skip if already exists)
echo "[2/4] Generating C code with im2col..."
if [ ! -f "$BUILD_DIR/${MODEL_NAME}_with_im2col.c" ]; then
    $BUILD_DIR/onnx2c -p im2col -l 0 "$MODEL" > "$BUILD_DIR/${MODEL_NAME}_with_im2col.c" 2>/dev/null
fi

# Compile (skip if binaries exist)
echo "[3/4] Compiling..."
if [ ! -f "$BUILD_DIR/${MODEL_NAME}_no_im2col" ] || [ ! -f "$BUILD_DIR/${MODEL_NAME}_with_im2col" ]; then
    cat > "$BUILD_DIR/benchmark_wrapper.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern void entry(const float* input, float* output);

int main(int argc, char** argv) {
    int iterations = 100;
    if (argc > 1) iterations = atoi(argv[1]);
    
    float* input = malloc(1 * 3 * 224 * 224 * sizeof(float));
    float* output = malloc(1 * 1000 * sizeof(float));
    
    for (int i = 0; i < 1 * 3 * 224 * 224; i++) input[i] = 0.0f;
    
    for (int i = 0; i < 3; i++) entry(input, output);
    
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

    echo "      Compiling without im2col..."
    gcc -O3 -o "$BUILD_DIR/${MODEL_NAME}_no_im2col" "$BUILD_DIR/${MODEL_NAME}_no_im2col.c" "$BUILD_DIR/benchmark_wrapper.c" -lm
    
    echo "      Compiling with im2col..."
    gcc -O3 -o "$BUILD_DIR/${MODEL_NAME}_with_im2col" "$BUILD_DIR/${MODEL_NAME}_with_im2col.c" "$BUILD_DIR/benchmark_wrapper.c" -lm
fi

# Run benchmarks
ITERATIONS=${1:-10}
echo "[4/4] Running benchmarks ($ITERATIONS iterations)..."
echo ""

echo "--------------------------------------------"
echo " WITHOUT im2col:"
echo "--------------------------------------------"
RESULT_NO_IM2COL=$($BUILD_DIR/${MODEL_NAME}_no_im2col $ITERATIONS)
echo "$RESULT_NO_IM2COL"

echo ""
echo "--------------------------------------------"
echo " WITH im2col:"
echo "--------------------------------------------"
RESULT_WITH_IM2COL=$($BUILD_DIR/${MODEL_NAME}_with_im2col $ITERATIONS)
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
