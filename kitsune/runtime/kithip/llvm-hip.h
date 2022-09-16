#ifndef _KITSUNE_LLVM_GPU_ABI_HIP_H_
#define _KITSUNE_LLVM_GPU_ABI_HIP_H_

#include<llvm/Target/TargetMachine.h>

void* hipManagedMalloc(size_t n);
int initHIP();
void* launchHIPKernel(llvm::Module& m, void** args, size_t n);
void waitHIPKernel(void* wait);

#endif
