# Kitsune support
kitsune_install_prefix:=${CMAKE_INSTALL_PREFIX}

# Cuda support
kitsune_cuda_enable:="${KITSUNE_CUDA_ENABLED}"
ifeq ($(kitsune_cuda_enable),"ON")
  $(info config: cuda target enabled.)
  KITSUNE_CUDA_ENABLED:=true
endif

# Hip support
kitsune_hip_enable:="${KITSUNE_HIP_ENABLED}"
ifeq ($(kitsune_hip_enable),"ON")
  $(info config: hip target enabled.)
  KITSUNE_HIP_ENABLED:=true
  ROCM_PATH:=${ROCM_PATH}
endif

# Kokkos support
kitsune_kokkos_enable:="${KITSUNE_KOKKOS_ENABLED}"
ifeq ($(kitsune_kokkos_enable),"ON")
  $(info config: kokkos codegen enabled.)
  KITSUNE_KOKKOS_ENABLED:=true
endif

