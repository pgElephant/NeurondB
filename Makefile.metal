# Metal-specific compilation rules
# Compile Metal implementation with Objective-C
src/gpu/gpu_metal_impl.o: src/gpu/gpu_metal_impl.m src/gpu/gpu_metal_wrapper.h
	$(CC) $(CFLAGS) $(PG_CPPFLAGS) -c -o $@ $<

# Clean Metal objects
clean-metal:
	rm -f src/gpu/gpu_metal.o src/gpu/gpu_metal_impl.o

.PHONY: clean-metal
