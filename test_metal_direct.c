#include <stdio.h>
#include "src/gpu/gpu_metal_wrapper.h"

int main() {
    printf("Testing Metal backend directly...\n");
    
    if (metal_backend_init()) {
        printf("✅ Metal init SUCCESS!\n");
        printf("Device: %s\n", metal_backend_device_name());
        
        float a[] = {1.0, 2.0, 3.0, 4.0};
        float b[] = {4.0, 3.0, 2.0, 1.0};
        
        float dist = metal_backend_l2_distance(a, b, 4);
        printf("L2 Distance: %f\n", dist);
        
        metal_backend_cleanup();
    } else {
        printf("✗ Metal init FAILED\n");
    }
    
    return 0;
}
