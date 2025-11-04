#!/bin/bash
# Compile Metal with proper include order to avoid conflicts

/usr/bin/clang \
    -c -o src/gpu/gpu_metal_impl.o \
    src/gpu/gpu_metal_impl.m \
    -fPIC -O2 \
    -DNS_BLOCK_ASSERTIONS \
    -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Metal.framework/Headers \
    -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/MetalPerformanceShaders.framework/Headers \
    -framework Metal \
    -framework MetalPerformanceShaders \
    -framework Foundation \
    -Wno-everything

echo "Metal compilation status: $?"
ls -lh src/gpu/gpu_metal_impl.o 2>/dev/null && echo "✅ SUCCESS!" || echo "✗ FAILED"
