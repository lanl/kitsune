include ../common.mk
include ../cuda.mk
include ../kokkos.mk

# General (shared) flags (see common.mk)
clang_flags=${cxx_flags} ${opt_flags} ${clang_info_flags}

#### Kitsune+Tapir options 
#  Notes:
#   -ftapir=cuda will target the CUDA ABI transform. 
#   -cuabi-verbose dumps kernel stats via ptxas output. 
#   -cuabi-keep-files saves intermediate files (.ll, .ptx, .fatbin) on disk for a closer look at codegen details. 
#
tapir_cu_flags=-ftapir=cuda -ffp-contract=fast -mllvm -cuabi-arch=${NVARCH} -I${CUDA_HOME}/include 
tapir_stripmine_flags=-mllvm -stripmine-count=1 -mllvm -stripmine-coarsen-factor=1 -mllvm -cuabi-default-grainsize=1
####

kokkos_cu_flags=-I${kokkos_cu_prefix}/include ${nvcc_cxx_flags} -m64 -arch=${NVARCH} --resource-usage -O3 --no-exceptions -I${kitsune_prefix}/include

all: saxpy-forall.cuda.${host_arch} saxpy-kokkos.${host_arch}

saxpy-forall.cuda.${host_arch}: saxpy-forall.cpp
	${clangxx} -ftime-report ${clang_flags} ${tapir_cu_flags} ${tapir_stripmine_flags} -o $@ $< -Xlinker -rpath=${kitsune_prefix}/lib

saxpy-kokkos.${host_arch}: saxpy-kokkos.cpp
	${nvcc_wrapper} ${kokkos_cu_flags} -o $@ $< ${kokkos_ld_flags} ${kokkos_ld_libs} -Xlinker -rpath=${kokkos_cu_prefix}/lib

clean:
	-rm -f *.${host_arch}
	-rm -f *.fbin *.s *.ptx *.ll 
	-rm -rf *-cfg-tmp
	-rm -f *.bc
	-rm -f *.fatbin
	-rm -f *.csv *.log *.s 
