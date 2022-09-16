#include"llvm-gpu.h"
#include"llvm-cuda.h"
#include"llvm-hip.h"
#include"llvm-spirv.h"
#include<llvm/IR/Module.h>
#include<llvm/IRReader/IRReader.h>
#include<llvm/Support/SourceMgr.h>
#include<fstream>
#include<cassert>
#include<error.h>
#include<stdbool.h>

void err(const char* msg){
  return error(1, 1, "%s", msg);
}

typedef enum {
  none,
  spirv,
  hip,
  cuda
} runtime;

static runtime globalRuntime = none;

extern "C" 
void *gpuManagedMalloc(uint64_t n){
  if (globalRuntime == none)
    initRuntime();
  switch(globalRuntime){
    case hip:
      return hipManagedMalloc(n);
    case cuda:
      return __kitrt_cuMemAllocManaged(n);
    default:
      err("no spirv managed malloc");
  }
  return NULL;
}

extern "C" 
void initRuntime() {
  if (globalRuntime != none) return;

  if (__kitrt_cuInit()) {
    globalRuntime = cuda;
    return;
  }

  if(initHIP()) {
    globalRuntime = hip;
    return;
  }

  if(initSPIRV()) {
    globalRuntime = spirv;
    return;
  }

  err("No suitable gpu hardware found, failed to initialize kitsune GPU runtime.\n");
}

extern "C"
void* launchBCKernel(const char* bc, uint64_t bcsize, void** args, uint64_t n) {
  llvm::LLVMContext C;
  llvm::SMDiagnostic SMD;
  //std::string strbuf(bc, bcsize);
  //std::ofstream out("runtime.bc");
  //out << strbuf;
  //out.close();

  llvm::StringRef sr(bc, bcsize);
  llvm::MemoryBufferRef mbr(sr, "kernelModRef");
  std::unique_ptr<llvm::Module> mod = parseIR(mbr, SMD, C);
  if(!mod){
    SMD.print("Failed to parse kernel IR: ", llvm::errs());
    exit(1);
  }

  return launchKernel(*mod, args, n);
}

extern "C"
void* launchKernel(llvm::Module& bc, void** args, uint64_t n) {
  switch(globalRuntime){
    case spirv:
      return launchSPIRVKernel(bc, args, n);
    case hip:
      return launchHIPKernel(bc, args, n);
    case cuda:
      return __kitrt_cuLaunchKernel(bc, args, n);
    default:
      err("can't launch kernel -- invalid/uninitialized runtime instance.");
  }
  return NULL;
}

extern "C"
void waitKernel(void* wait) {

  switch(globalRuntime) {
    case spirv:
      return waitSPIRVKernel(wait);
    case hip:
      return waitHIPKernel(wait);
    case cuda:
      return __kitrt_cuStreamSynchronize(wait);
    default:
      err("can't wait for kernel -- invalid/uninitialized runtime instance.");
  }
}
