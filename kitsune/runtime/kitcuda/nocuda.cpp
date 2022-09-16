#include<stdlib.h>
#include<stdbool.h>
#include<llvm/IR/Module.h>

void* cudaManagedMalloc(size_t n){return NULL;}
int initCUDA(){return false; }
void* launchCUDAKernel(llvm::Module& m, void** args, size_t n){return NULL;}
void waitCUDAKernel(void* wait) {}
