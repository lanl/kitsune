exes=l2_opencilk l2_gpu

all: $(exes)

%_serial: %.c
	kitcc $< -o $@ -O2 -fuse-ld=lld -fPIC -ftapir=serial -lm -lllvm-gpu -lomp #-mllvm -debug-only="loop-stripmine"

%_opencilk: %.c
	kitcc $< -o $@ -O2 -fuse-ld=lld -fPIC -ftapir=opencilk -lm -lllvm-gpu -lomp #-mllvm -debug-only="loop-stripmine"

%_opencilk.ll: %.c
	kitcc $< -o $@ -O2 -fPIC -ftapir=opencilk -mllvm -debug-only="loop-stripmine" -S -emit-llvm

%_gpu: %.c
	kitcc $< -o $@ -O2 -fPIC -ftapir=gpu -lm -lllvm-gpu -lomp -mllvm -debug-only="gpuabi"

%_gpu.ll: %.c
	kitcc $< -o $@ -O2 -fPIC -ftapir=gpu -mllvm -debug-only="loop-stripmine" -S -emit-llvm

%_cuda: %.c
	kitcc $< -o $@ -O2 -fPIC -ftapir=cuda -lm -lllvm-gpu -lomp -v -mllvm -cuabi-arch=sm_75

%_stripmined.ll: %.c 
	kitcc $< -o $@ -O1 -ftapir=none -S -emit-llvm -mllvm -stripmine-loop

%_openmp: %_openmp.c
	kitcc $< -fopenmp -O2 -o $@ -lm 

clean: 
	rm -f $(exes) l2_stripmined.ll l2 l2.ll
