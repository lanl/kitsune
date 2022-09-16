#include<stdbool.h>
#include<dlfcn.h>
#include<llvm/IR/Module.h>

bool initSPIRV();
void* launchSPIRVKernel(llvm::Module& kernel, void**args, size_t n);
void waitSPIRVKernel(void* wait);
