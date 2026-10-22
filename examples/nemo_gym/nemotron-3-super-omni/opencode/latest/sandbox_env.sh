#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Source after setting the three sandbox dependency paths.
for artifact in "$OPENCODE_SANDBOX_BWRAP_ROOT/lib/ld-linux-aarch64.so.1" \
  "$OPENCODE_SANDBOX_BWRAP_ROOT/root/usr/bin/bwrap" \
  "$OPENCODE_TOOL_PYTHON/bin/python" "$OPENCODE_TOOL_PYTHON/pyvenv.cfg" \
  "$OPENCODE_PLUGIN_DEPS/node_modules/@opencode-ai/plugin/package.json"; do
  test -f "$artifact" || { echo "Missing episode sandbox dependency: $artifact" >&2; exit 1; }
done
sandbox_python_root=$(dirname -- "$(dirname -- "$(realpath -- "$OPENCODE_TOOL_PYTHON/bin/python")")")
sandbox_python_home=$(awk -F ' = ' '$1 == "home" {print $2}' "$OPENCODE_TOOL_PYTHON/pyvenv.cfg")
[[ $sandbox_python_home == */bin ]] || { echo "Invalid tool Python home" >&2; exit 1; }
sandbox_python_home=${sandbox_python_home%/bin}
OPENCODE_SANDBOX_MOUNTS=""
for directory in "$OPENCODE_SANDBOX_BWRAP_ROOT" "$OPENCODE_TOOL_PYTHON" \
  "$OPENCODE_PLUGIN_DEPS" "$sandbox_python_root" "$sandbox_python_home"; do
  test -d "$directory" || { echo "Missing episode sandbox directory: $directory" >&2; exit 1; }
  OPENCODE_SANDBOX_MOUNTS+=",$directory:$directory:ro"
done
export OPENCODE_SANDBOX_MOUNTS
