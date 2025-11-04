#!/bin/bash
# Final Metal compilation with proper SDK paths

SDK="/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
TOOLCHAIN="/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain"

/usr/bin/clang \
    -c -o src/gpu/gpu_metal_impl.o \
    src/gpu/gpu_metal_impl.m \
    -fPIC -O2 \
    -isysroot "$SDK" \
    -I"$TOOLCHAIN/usr/lib/clang/16/include" \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -framework Foundation \
    -Wno-everything \
    2>&1

status=$?
echo "═══════════════════════════════════════════════════════════════════"
if [ $status -eq 0 ]; then
    echo "✅ Metal GPU implementation compiled successfully!"
    ls -lh src/gpu/gpu_metal_impl.o
    echo ""
    echo "Metal symbols exported:"
    nm -gU src/gpu/gpu_metal_impl.o | grep "metal_backend" | head -8
else
    echo "✗ Compilation failed with exit code: $status"
fi
echo "═══════════════════════════════════════════════════════════════════"
exit $status
