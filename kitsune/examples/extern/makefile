cc=clang
cxx=clang++

all: extern

# Builds LLVM IR for library
add.o: add.c
	${cc} -c -flto $< -O1 -o $@

# Builds LLVM IR for our main module 
extern.o: extern.cpp
	${cxx} -c -fno-exceptions -flto -O1 -ftapir=gpu $< -o $@

# Links them together, then performs Tapir lowering
extern: add.o extern.o
	${cxx} --verbose -fuse-ld=lld -Wl,--tapir-target=gpu,-mllvm,-debug-pass=Arguments,--lto-debug-pass-manager,--lto-O2 -flto -ftapir=gpu $^ -O1 -o  $@ 
# ${cxx} --verbose -fuse-ld=lld -Wl,-mllvm,-print-after-all,--lto-legacy-pass-manager,--tapir-target=gpu,-mllvm,-debug-pass=Arguments,--lto-debug-pass-manager,--lto-O2 -flto $^ -O1 -o  $@ 

clean: 
	rm -f add.o extern.o extern
