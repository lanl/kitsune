include ../common.mk
include ../cuda.mk

# General (shared) flags (see common.mk)
clang_flags=${c_flags} ${opt_flags} ${clang_info_flags}


#### Tapir parameters 
COARSEN_FACTOR?=1
GRAINSIZE?=1
THREADS_PER_BLOCK?=256
BLOCKS_PER_GRID?=0

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
tapir_cu_flags=-mllvm --debug-only=cuda-abi -ftapir=cuda -ffp-contract=fast -mllvm -cuabi-keep-files -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_stripmine_flags=-mllvm -stripmine-count=${GRAINSIZE} -mllvm -stripmine-coarsen-factor=${COARSEN_FACTOR} -mllvm -cuabi-default-grainsize=${GRAINSIZE}
####

all: 3D.${host_arch} 3D-forall.${host_arch}

3D.${host_arch}: 3D.c
	${clang} ${clang_flags} -fopenmp -o $@ $<  -lm 

3D-forall.${host_arch}: 3D-forall.c
	${clang} ${cflags} ${clang_flags} ${tapir_cu_flags} -I${kitsune_prefix}/include -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib -lm 

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.ll *.ptx *.csv *.log
