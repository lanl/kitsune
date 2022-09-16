#include<stdlib.h>
#include<stdbool.h>
#include<llvm/IR/Module.h>

bool initSPIRV(){ return false; }
void* launchSPIRVKernel(llvm::Module& kernel, void** args, size_t n) {return NULL;}
void waitSPIRVKernel(void* wait) {}
