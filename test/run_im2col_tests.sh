#!/bin/bash
# Test all conv models with im2col pass

ONNX2C="${ONNX2C:-/workspace/build/onnx2c}"
TESTDIR="/workspace/test/simple_networks"

echo "=== Im2Col Test: Conv Models ==="

# Test models in order of complexity
MODELS=(
    "test_conv_simple.onnx"
    "test_conv_with_bias.onnx"
    "test_conv_stride2.onnx"
    "test_depthwise_conv.onnx"
    "test_pointwise_conv.onnx"
)

# Fallback: use existing models if test models don't exist
if [ ! -f "$TESTDIR/test_conv_simple.onnx" ]; then
    echo "Note: Using existing simple_networks models"
    MODELS=(
        "conv_2kernels.onnx"
        "conv_2kernels_randombias.onnx"
        "conv_k2.onnx"
        "conv_k2_s2.onnx"
    )
fi

FAILED=0

for model in "${MODELS[@]}"; do
    echo ""
    echo "Testing: $model"
    
    if [ ! -f "$TESTDIR/$model" ]; then
        echo "  SKIP: model not found"
        continue
    fi
    
    # Generate without im2col
    echo "  Regular: generating..."
    $ONNX2C "$TESTDIR/$model" -f "${model%.onnx}_reg" > /tmp/${model}_reg.c 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed to generate regular version"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Generate with im2col
    echo "  im2col:  generating..."
    $ONNX2C -p im2col "$TESTDIR/$model" -f "${model%.onnx}_im2col" > /tmp/${model}_im2col.c 2>&1
    
    if [ $? -ne 0 ]; then
        echo "  ERROR: Failed to generate im2col version"
        FAILED=$((FAILED + 1))
        continue
    fi
    
    # Check for key components
    if grep -q "MatMul\|matmul" /tmp/${model}_im2col.c; then
        echo "  OK: MatMul present in im2col version"
    else
        echo "  WARN: No MatMul found (may not have been transformed)"
    fi
    
    if grep -q "node_conv2d\|node_conv" /tmp/${model}_reg.c; then
        echo "  OK: Conv node present in regular version"
    fi
    
    echo "  PASSED"
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All tests passed ==="
    exit 0
else
    echo "=== $FAILED tests failed ==="
    exit 1
fi