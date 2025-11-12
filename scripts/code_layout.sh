#!/bin/bash
# Run clang-format againts the entire code base.
# Don't run this, rather run ./scripts/code_layout_diff.sh
# instead!

clang-format -i $(find src -name "*.cc" -or -name "*.h") --verbose 
