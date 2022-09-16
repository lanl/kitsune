include ../common.mk
include ../cuda.mk
include ../kokkos.mk
SHELL:=/bin/bash 

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}
# kitsune kokkos mode flags.
kit_kokkos_flags=-fkokkos -fkokkos-no-init


#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_launch_flags=-I${CUDA_HOME}/include
tapir_cu_flags=-ffast-math -ftapir=cuda -mllvm -cuabi-verbose -mllvm -cuabi-arch=${NVARCH} ${tapir_cu_launch_flags} 
tapir_stripmine_flags=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1  -mllvm -cuabi-default-grainsize=1
####

#### CUDA flags
nvcc_flags = -allow-unsupported-compiler --std c++17 -m64 -arch ${NVARCH} --resource-usage --optimize 3 --no-exceptions -I${kitsune_prefix}/include
nvcc_kokkos_flags=${nvcc_flags} -I${kokkos_cuda_prefix}/include -I${kitsune_prefix}/include
clang_cuda_flags = -xcuda --cuda-gpu-arch=${NVARCH} ${cxx_flags} ${opt_flags} -I${kitsune_prefix}/include
####

### Kokkos stuff 
kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include

all: vecadd-forall.cuda.${host_arch} \
     vecadd-forall.cuda.${host_arch} \
     vecadd.nvcc.${host_arch} \
     vecadd.clang.${host_arch} \
     vecadd-kokkos.${host_arch}

# Kitsune-Tapir versions.
vecadd-forall.cuda.${host_arch}: vecadd-forall.cpp
	time ${clangxx} ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

# Cuda version.
vecadd.nvcc.${host_arch}: vecadd.cu
	time ${nvcc} ${nvcc_flags} -o $@ $<
vecadd.clang.${host_arch}: vecadd.cu
	time ${clangxx} ${clang_cuda_flags} -o $@ $< -L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread

# Kokkos version. 
vecadd-kokkos.${host_arch}: vecadd-kokkos.cpp
	time ${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fatbin
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.ll *.ptx *.csv *.log *.fbin *.s
	
