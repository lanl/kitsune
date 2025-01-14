exes=l2 l2_gpu l2_serial l2_openmp
all:${exes}

l2.ll : l2.c
	clang -S -emit-llvm l2.c -O1 -ftapir=none 

l2_stripmined.ll: l2.ll
	opt -S -passes="loop-stripmine" -o l2_stripmined.ll l2.ll 

l2_stripmined_opt.ll: l2_stripmined.ll
	opt -S -O2 -o l2_stripmined_opt.ll l2_stripmined.ll

l2: l2_stripmined_opt.ll
	clang -ftapir=opencilk -O1 l2_stripmined_opt.ll -o l2 -lm  -lllvm-gpu -lpthread -lomp

l2_gpu.ll: l2.ll
	clang -ftapir=gpu -O1 -S -emit-llvm $< -o $@ -mllvm --debug-only=gpuabi

l2_gpu: l2_gpu.ll
	clang l2_gpu.ll -fPIC -O1 -o l2_gpu -ftapir=gpu -lm -lllvm-gpu -lpthread -lomp

l2_serial: l2.ll
	clang l2.ll -o l2_serial -ftapir=serial -lllvm-gpu -lm -lomp

l2_openmp: l2_openmp.c
	clang l2_openmp.c -fopenmp -O2 -o l2_openmp -lm

clean: 
	rm -f ${exes} l2.ll
