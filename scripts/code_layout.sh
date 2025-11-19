#!/bin/bash

clang-format -i $(find src -name "*.cc" -or -name "*.h") --verbose 
