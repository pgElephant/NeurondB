#!/bin/bash
# Compile Metal with SDK-only includes to avoid /usr/local/include conflicts

SDK_PATH="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

/usr/bin/clang \
    -c -o src/gpu/gpu_metal_impl.o \
    src/gpu/gpu_metal_impl.m \
    -fPIC -O2 \
    -DNS_BLOCK_ASSERTIONS \
    -nostdinc \
    -isystem "$SDK_PATH/usr/include" \
    -isystem "$SDK_PATH/usr/include/c++/v1" \
    -F "$SDK_PATH/System/Library/Frameworks" \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -framework Foundation \
    -Wno-everything \
    2>&1

status=$?
echo "═══════════════════════════════════════════════════════════════════"
if [ $status -eq 0 ]; then
    echo "✅ Metal compilation SUCCESS!"
    ls -lh src/gpu/gpu_metal_impl.o
    nm src/gpu/gpu_metal_impl.o | grep "metal_backend" | head -5
else
    echo "✗ Metal compilation FAILED (exit: $status)"
fi
echo "═══════════════════════════════════════════════════════════════════"
