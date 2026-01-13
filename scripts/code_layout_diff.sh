#!/usr/bin/env bash
# Format the diff using clang-format.
# Run against committed changes from BASE_BRANCH
# as well as format the unstaged changes.

set -euo pipefail

BASE_BRANCH=${1:-origin/master}

{
  git diff -U0 "$BASE_BRANCH"...HEAD
  git diff -U0
} | clang-format-diff -p1 -i

