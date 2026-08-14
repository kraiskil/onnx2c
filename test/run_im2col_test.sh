#!/bin/bash
# Automated test comparing regular conv vs im2col conv using onnx2c

set -e

ONNX2C="${ONNX2C:-/workspace/build/onnx2c}"
TESTDIR="/workspace/test/simple_networks"
OUTDIR="/workspace/test/im2col_output"

mkdir -p "$OUTDIR"

# Test models
MODELS=(
    "conv_2kernels.onnx"
    "conv_2kernels_randombias.onnx"
    "conv_k2.onnx"
    "conv_k2_s2.onnx"
)

echo "=== Im2Col Automated Tests ==="
echo

for model in "${MODELS[@]}"; do
    echo "Testing: $model"
    
    base="${model%.onnx}"
    
    # Generate without im2col
    $ONNX2C "$TESTDIR/$model" -f "${base}_regular" > "$OUTDIR/${base}_regular.c" 2>/dev/null
    
    # Generate with im2col
    $ONNX2C -p im2col "$TESTDIR/$model" -f "${base}_im2col" > "$OUTDIR/${base}_im2col.c" 2>/dev/null
    
    # Compare C code - check key transformations exist
    if grep -q "node.*_matmul_mm" "$OUTDIR/${base}_im2col.c"; then
        echo "  im2col: MatMul found"
    else
        echo "  ERROR: No MatMul in im2col version"
        exit 1
    fi
    
    if ! grep -q "node_conv2d(" "$OUTDIR/${base}_regular.c"; then
        echo "  ERROR: Regular version should have direct conv"
        exit 1
    fi
    
    echo "  PASSED"
    echo
done

echo "=== All tests passed ==="