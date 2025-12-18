exes=l2_opencilk l2_gpu triangle_opencilk triangle_gpu triangle_serial

all: $(exes)

%_serial: %.c
	clang $< -o $@ -O2 -fuse-ld=lld -fPIC -ftapir=serial -lm -lllvm-gpu -lomp #-mllvm -debug-only="loop-stripmine"

%_opencilk: %.c
	clang $< -o $@ -O2 -fuse-ld=lld -fPIC -ftapir=opencilk -lm -lllvm-gpu -lomp #-mllvm -debug-only="loop-stripmine"

%_opencilk.ll: %.c
	clang $< -o $@ -O2 -fPIC -ftapir=opencilk -mllvm -debug-only="loop-stripmine" -S -emit-llvm

%_gpu: %.c
	clang $< -o $@ -O2 -fPIC -ftapir=gpu -lm -lllvm-gpu -lomp #-mllvm -debug-only="loop-stripmine"

%_gpu.ll: %.c
	clang $< -o $@ -O2 -fPIC -ftapir=gpu -mllvm -debug-only="loop-stripmine" -S -emit-llvm

%_cuda: %.c
	kitcc $< -o $@ -O2 -fPIC -ftapir=cuda -lm -lllvm-gpu -lomp -v -mllvm -cuabi-arch=sm_75

%_stripmined.ll: %.c 
	clang $< -o $@ -O1 -ftapir=none -S -emit-llvm -mllvm -stripmine-loop

%_openmp: %_openmp.c
	clang $< -fopenmp -O2 -o $@ -lm 

clean: 
	rm -f $(exes) l2_stripmined.ll l2 l2.ll
