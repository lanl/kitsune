exes=l2_serial l2_opencilk l2_gpu l2_serial l2_openmp l2_cuda

all: $(exes)

l2_serial: l2.c
	clang $< -o $@ -O2 -fuse-ld=lld -fPIC -o l2_serial -ftapir=serial -lm -lllvm-gpu -lomp

l2_opencilk: l2.c
	clang $< -o $@ -O2 -fuse-ld=lld -fPIC -o l2_opencilk -ftapir=opencilk -lm -lllvm-gpu -lomp -mllvm -debug-only="loop-stripmine"

l2_gpu: l2.c
	clang $< -o $@ -O2 -fPIC -o l2_gpu -ftapir=gpu -lm -lllvm-gpu -lomp -mllvm -debug-only="loop-stripmine"

l2_cuda: l2.c
	kitcc $< -o $@ -O2 -fPIC -o l2_cuda -ftapir=cuda -lm -lllvm-gpu -lomp -v -mllvm -cuabi-arch=sm_75

l2_openmp: l2_openmp.c
	clang l2_openmp.c -fopenmp -O2 -o l2_openmp -lm -lgomp

clean: 
	rm -f $(exes) l2_stripmined.ll l2 l2.ll
