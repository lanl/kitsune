include ../common.mk
include ../cuda.mk
include ../kokkos.mk

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}


#### Tapir parameters 
COARSEN_FACTOR?=1
GRAINSIZE?=1
THREADS_PER_BLOCK?=256
BLOCKS_PER_GRID?=0

$(info coarsen factor    : ${COARSEN_FACTOR})
$(info grainsize         : ${GRAINSIZE})
$(info blocks per grid   : ${BLOCKS_PER_GRID})
$(info threads per block : ${THREADS_PER_BLOCK})

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
#tapir_cu_launch_flags=-mllvm -cuabi-threads-per-block=${THREADS_PER_BLOCK} -mllvm -cuabi-blocks-per-grid=${BLOCKS_PER_GRID}
tapir_cu_flags=-ffast-math -ftapir=cuda -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags}
tapir_stripmine_flags=-mllvm -stripmine-count=${GRAINSIZE} -mllvm -stripmine-coarsen-factor=${COARSEN_FACTOR}  -mllvm -cuabi-default-grainsize=${GRAINSIZE}
####


%.o: %.cpp 
	${clangxx} -c ${clang_flags} ${tapir_cu_flags} $< -o $@

# Kitsune-Tapir versions.
objs=main.o init.o compute.o
multi-module-test.${host_arch}: ${objs}
	${clangxx} ${clang_flags} ${tapir_cu_flags} -o $@ ${objs} -Xlinker -rpath=${kitsune_prefix}/lib

clean:
	-rm -f *.${host_arch} *.o 
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc

realclean: clean
	-rm -f *.ll *.ptx *.csv *.log *.fatbin *.o 
	
