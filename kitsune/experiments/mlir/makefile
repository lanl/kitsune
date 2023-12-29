all: linalg-tapir linalg-serial linalg-omp 

scf.mlir: linalg.mlir
	mlir-opt $< -o $@ \
		 -convert-linalg-to-parallel-loops  \
		 -lower-affine 
 
tapir.mlir: scf.mlir 
	mlir-opt $< -o $@ \
		-convert-scf-to-tapir \
		-convert-scf-to-cf 

defunc.mlir: tapir.mlir
	mlir-opt $< -o $@ \
		-pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm)" 

nomemref.mlir: defunc.mlir
	mlir-opt $< -o $@ \
		-convert-memref-to-llvm \
		-reconcile-unrealized-casts

linalg.ll: nomemref.mlir
	mlir-translate $< -o $@ \
		-mlir-to-llvmir

linalg01.ll: linalg.ll
	opt -O2 -S $< -o $@

linalg-stripmined.ll: linalg01.ll
	opt -passes=loop-stripmine -S $< -o $@

linalg.o: linalg01.ll
	clang -g -ftapir=opencilk -O2 -c $< -o $@
	
linalg-tapir: test.c linalg.o
	clang -g -O2 $^ -o $@ /opt/kitsune/lib/clang/16/lib/darwin/libopencilk_osx.a

omp.mlir: scf.mlir 
	mlir-opt $< -o $@ \
		-convert-scf-to-openmp  \
		-convert-scf-to-cf  

defunc-omp.mlir: omp.mlir
	mlir-opt $< -o $@ \
		-pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm)" 

nomemref-omp.mlir: defunc-omp.mlir
	mlir-opt $< -o $@ \
		-convert-memref-to-llvm \
		-reconcile-unrealized-casts

linalg-omp.ll: nomemref-omp.mlir
	mlir-translate $< -o $@ \
		-mlir-to-llvmir

linalg01-omp.ll: linalg-omp.ll
	opt -O1 -S $< -o $@

linalg-omp.o: linalg01-omp.ll
	clang -g -fopenmp -O2 -c $< -o $@

linalg-omp: test.c linalg-omp.o
	clang -g -O2 $^ -o $@ /opt/kitsune/lib/clang/16/lib/darwin/libopencilk_osx.a

serial.mlir: scf.mlir 
	mlir-opt $< -o $@ \
		-convert-scf-to-cf  

defunc-serial.mlir: serial.mlir
	mlir-opt $< -o $@ \
		-pass-pipeline="builtin.module(func.func(convert-math-to-llvm,convert-arith-to-llvm),convert-func-to-llvm)" 

nomemref-serial.mlir: defunc-serial.mlir
	mlir-opt $< -o $@ \
		-convert-memref-to-llvm \
		-reconcile-unrealized-casts

linalg-serial.ll: nomemref-serial.mlir
	mlir-translate $< -o $@ \
		-mlir-to-llvmir

linalg01-serial.ll: linalg-serial.ll
	opt -O1 -S $< -o $@

linalg-serial.o: linalg01-serial.ll
	clang -g -fopenmp -O2 -c $< -o $@

linalg-serial: test.c linalg-serial.o
	clang -g -O2 $^ -o $@ /opt/kitsune/lib/clang/16/lib/darwin/libopencilk_osx.a
