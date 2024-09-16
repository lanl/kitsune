all: l2 l2_gpu l2_serial

l2.ll : l2.c
	clang -S -emit-llvm l2.c -O1 -ftapir=none 

l2_stripmined.ll: l2.ll
	opt -S -passes="loop-stripmine" -o l2_stripmined.ll l2.ll 

l2_stripmined_opt.ll: l2_stripmined.ll
	opt -S -O2 -o l2_stripmined_opt.ll l2_stripmined.ll

l2: l2_stripmined_opt.ll
	clang -ftapir=opencilk -O2 l2_stripmined_opt.ll -o l2 -lm  -lllvm-gpu

l2_gpu.ll: l2_stripmined_opt.ll
	clang -ftapir=gpu -O1 -S -emit-llvm $< -o $@ 

l2_gpu: l2_gpu.ll
	clang l2_gpu.ll -fPIC -o l2_gpu -ftapir=gpu -lm -lllvm-gpu

l2_serial: l2.ll
	clang l2.ll -o l2_serial -ftapir=serial -lllvm-gpu -lm

clean: 
	rm -f l2_stripmined.ll l2 l2.ll
