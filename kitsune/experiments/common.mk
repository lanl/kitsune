#
# Set a host-architecture-centric install path for the
# full kitsune toolchain -- can also be set via the
# user's enviornment with KITSUNE_PREFIX.
#
host_arch := $(shell uname -o -m | awk '{print $$1}')
ifeq ($(KITSUNE_PREFIX),)
  kitsune_prefix=/projects/kitsune/13.x/${host_arch}
else
  kitsune_prefix=$(KITSUNE_PREFIX)
endif
$(info kitsune+tapir install prefix: ${kitsune_prefix})

c_flags=-I${kitsune_prefix}/include
cxx_flags=-std=c++17 -fno-exceptions -I${kitsune_prefix}/include
opt_flags=-O3 
clang_info_f2ags=

clang=${kitsune_prefix}/bin/clang
clangxx=${kitsune_prefix}/bin/clang++
opt=${kitsune_prefix}/bin/opt
