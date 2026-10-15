#!/usr/bin/env bash
# fscaa_validate.sh <extracted spv dir> - run every FSCAA transform over its target family and
# spirv-val every output. The extracted dir holds one folder per family (unzip of each .xsa).
# Prints one line per transform: patched / refused (by reason) / spirv-val valid / INVALID.
# SPDX-License-Identifier: GPL-3.0-or-later
set -uo pipefail
SRC="$1"
OUT="${TEMP:-/tmp}/fscaa_val"
V=/c/VulkanSDK/1.4.357.0/Bin/spirv-val.exe
cd "$(dirname "$0")/.."
mkdir -p "$OUT"
g++ -O2 -std=c++17 tools/fscaa_patch_test.cpp -I src/vklayer -o "$OUT/fscaa_patch_test.exe" || { echo "BUILD FAILED"; exit 1; }
rc=0
for job in tint:line3d wire:line3d light:light light:ground_lights specular:deferred_gbuf alpha:terrain; do
  L=${job%%:*}; F=${job##*:}; D="$OUT/$L-$F"; rm -rf "$D"; mkdir -p "$D"
  find "$SRC/$F" -name '*.spv' -print0 | xargs -0 -n 500 "$OUT/fscaa_patch_test.exe" "$L" "$D" > "$D.log"
  ok=0; bad=0
  for f in "$D"/*.spv; do
    [ -e "$f" ] || continue
    if "$V" --target-env vulkan1.3 "$f" > "$f.val" 2>&1; then ok=$((ok+1)); rm -f "$f.val"; else bad=$((bad+1)); rc=1; fi
  done
  reasons=$(grep '^REFUSED' "$D.log" | sed -E 's/^REFUSED [^:]+: //' | sort | uniq -c | sed -E 's/^ +//' | tr '\n' ';')
  echo "$L on $F: $(grep -c '^OK' "$D.log") patched, $(grep -c '^REFUSED' "$D.log") refused [${reasons}] spirv-val $ok valid / $bad INVALID"
  [ "$bad" -gt 0 ] && ls "$D"/*.val | head -3 | while read v; do echo "   first error ($(basename "$v" .val)): $(head -2 "$v" | tr '\n' ' ' | cut -c1-300)"; done
done
exit $rc
