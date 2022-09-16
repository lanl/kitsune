ifeq ($(CUDA_HOME),)
  $(error "required environment variable CUDA_HOME not set!")
  ${error "   set it point to the install prefix of CUDA.")
  exit 1
endif

NVARCH?=sm_80

cuda_prefix=$(CUDA_HOME)

$(info "cuda settings:")
$(info "    install prefix: ${cuda_prefix}")
$(info "    default cuda architecture: ${NVARCH}")

nvcc=${CUDA_HOME}/bin/nvcc
ptxas=${CUDA_HOME}/bin/ptxas
fatbinary=${CUDA_HOME}/bin/fatbinary

nvcc_cxx_flags=-std=c++17 --no-exceptions --expt-extended-lambda -expt-relaxed-constexpr
nvcc_target_flags=-arch=${NVARCH}
clang_cu_flags=-xcuda --cuda-gpu_arch=${NVARCH}
