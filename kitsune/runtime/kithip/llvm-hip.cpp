#include<iostream>
#include<dlfcn.h>
#include<llvm/IR/LegacyPassManager.h>
#include<llvm/IR/Constants.h>
#include<llvm/IR/Instruction.h>
#include<llvm/IR/Instructions.h>
#include<llvm/IR/Intrinsics.h>
#include<llvm/IR/IntrinsicsAMDGPU.h>
#include<llvm/Transforms/Utils/BasicBlockUtils.h>
#include<llvm/IR/Verifier.h>
#include<llvm/Option/ArgList.h>
#include<llvm/Support/Program.h>
#include<llvm/Support/MemoryBuffer.h>
#include<llvm/Support/TargetSelect.h>
#include<llvm/Support/CommandLine.h>
#include<llvm/Support/raw_os_ostream.h>
#include<llvm/Target/TargetMachine.h>
#include<llvm/Support/ToolOutputFile.h>
#include<llvm/Support/TargetRegistry.h>
#include<llvm/Transforms/IPO/PassManagerBuilder.h>
#define __HIP_PLATFORM_HCC__ 1
#include<hip/hip_runtime.h>

#define declare(name) decltype(name)* name##_p = NULL
#define tryLoad(name) name##_p = (decltype(name)*)dlsym(hiphandle, #name)

void* hiphandle;
declare(hipGetDevice);
declare(hipGetDeviceCount);
declare(hipGetDeviceProperties);
declare(hipStreamCreate);
declare(hipModuleLoadData);
declare(hipModuleLaunchKernel);
declare(hipModuleGetFunction);
declare(hipGetErrorString);
declare(hipStreamSynchronize);
declare(hipStreamDestroy);
declare(hipInit);
hipError_t (*hipHostMalloc_p)(void** res, size_t n, int f);

extern "C" {

  void checkHIP(hipError_t in){
    if(in !=  HIP_SUCCESS){
      std::cerr << "Error: " << hipGetErrorString_p(in) << std::endl;
      exit(in);
    }
  }

  using namespace llvm;

  int initHIP(){
    if(hiphandle) return true;
    hiphandle = dlopen("libamdhip64.so", RTLD_LAZY);
    if(!hiphandle) return false;
    tryLoad(hipGetDevice);
    tryLoad(hipGetDeviceCount);
    tryLoad(hipGetDeviceProperties);
    tryLoad(hipStreamCreate);
    tryLoad(hipStreamDestroy);
    tryLoad(hipStreamSynchronize);
    tryLoad(hipModuleLoadData);
    tryLoad(hipModuleLaunchKernel);
    tryLoad(hipModuleGetFunction);
    tryLoad(hipInit);
    tryLoad(hipGetErrorString);
    hipHostMalloc_p = (decltype(hipHostMalloc_p))(dlsym(hiphandle, "hipHostMalloc"));
    hipInit_p(0);
    int count;
    hipGetDeviceCount_p(&count);
    if(count == 0) return false;
    return hiphandle != NULL;
  }

  void* hipManagedMalloc(size_t n){
    void* res;
    checkHIP(hipHostMalloc_p(&res, n, 0));
    return res;
  }

  void* launchHIPKernel(llvm::Module& m, void** args, size_t n) {
    LLVMContext& ctx = m.getContext();
    legacy::PassManager PM;
    legacy::FunctionPassManager FPM(&m);
    int deviceId;
    hipGetDevice_p(&deviceId);
    hipDeviceProp_t prop;
    hipGetDeviceProperties_p(&prop, deviceId);
    std::string gcnarch = "gfx" + std::to_string(prop.gcnArch);
    Triple TT("amdgcn", "amd", "amdhsa");
    m.setTargetTriple(TT.str());

    Function& F = *m.getFunction("kitsune_kernel");

    AttrBuilder Attrs;
    Attrs.addAttribute("target-cpu", gcnarch);
    //Attrs.addAttribute("target-features", cudafeatures + ",+" + cudaarch);
    /*
      Attrs.addAttribute(Attribute::NoRecurse);
      Attrs.addAttribute(Attribute::Convergent);
    */
    F.removeFnAttr("target-cpu");
    F.removeFnAttr("target-features");
    F.setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
    F.addAttributes(AttributeList::FunctionIndex, Attrs);

    auto tid = Intrinsic::getDeclaration(&m, Intrinsic::amdgcn_workitem_id_x);

    std::vector<std::pair<Instruction*, CallInst*>> tids;
    for(auto &BB : F){
      for(auto &I : BB){
        if(auto *CI = dyn_cast<CallInst>(&I)){
          if(Function *f = CI->getCalledFunction()){
            if(f->getName() == "gtid"){
              tids.push_back(std::make_pair(&I, CallInst::Create(tid)));
            }
          }
        }
      }
    }

    for(auto p : tids){
      ReplaceInstWithInst(p.first, p.second);
      p.second->setTailCall();
    }

    if(auto *f = m.getFunction("gtid")) f->eraseFromParent();

    m.print(llvm::errs(), nullptr);

    // Ugh, this sucks. Have to use command line utilities and temporary files
    // despite the code existing in the same repository. Might be worth looking
    // into how to do this in memory, though I'm not sure about linking.
    std::string ObjectFile = "/tmp/kernel.hip.o";
    std::string LinkedObjectFile = "/tmp/kernel.hip-l.o";
    std::string BundledObjectFile = "/tmp/kernel.hip-b.o";
    std::error_code EC;
    sys::fs::OpenFlags OpenFlags = sys::fs::F_None;
    std::unique_ptr<ToolOutputFile> FDOut =
        std::make_unique<ToolOutputFile>(ObjectFile, EC, OpenFlags);
    raw_pwrite_stream *fostr = &FDOut->os();

    std::string error;
    raw_os_ostream ostr(std::cout);
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();

    const Target *Target = TargetRegistry::lookupTarget("", TT, error);
    auto TargetMachine =
        Target->createTargetMachine(TT.getTriple(), gcnarch,
                                    "", TargetOptions(), Reloc::PIC_,
                                    CodeModel::Small, CodeGenOpt::Aggressive);
    m.setDataLayout(TargetMachine->createDataLayout());

    FPM.doInitialization();
    for (Function &F : m)
      FPM.run(F);
    FPM.doFinalization();
    //PM.add(createVerifierPass());
    bool Fail = TargetMachine->addPassesToEmitFile(
        PM, *fostr, nullptr,
        CodeGenFileType::CGFT_ObjectFile, false);
    assert(!Fail && "Failed to emit AMDGCN");
    // Add function optimization passes.
    PM.run(m);
    FDOut->keep();

    std::string clangOffloadBundle  = *sys::findProgramByName("clang-offload-bundler");
    std::string lld = *sys::findProgramByName("ld.lld");

    opt::ArgStringList offloadBundleArgList, lldArgList;
    std::string cpus = "-plugin-opt=mcpu=" + gcnarch;
    std::string lofs = "-o" + LinkedObjectFile;
    lldArgList.push_back(lld.c_str());
    lldArgList.push_back("-shared");
    lldArgList.push_back(cpus.c_str());
    lldArgList.push_back("-plugin-opt=-amdgpu-internalize-symbols");
    lldArgList.push_back("-plugin-opt=O3");
    lldArgList.push_back("-plugin-opt=-amdgpu-early-inline-all=true");
    lldArgList.push_back("-plugin-opt=-amdgpu-function-calls=false");
    lldArgList.push_back(lofs.c_str());
    lldArgList.push_back(ObjectFile.c_str());
    lldArgList.push_back(nullptr);

    auto lldsra = toStringRefArray(lldArgList.data());
    sys::ExecuteAndWait(lld, lldsra);

    // Warning: this changes to from hip- to hipv4- in llvm 13
    std::string targets = "-targets=host-x86_64-unknown-linux-gnu,hipv4-"
                          + m.getTargetTriple() + "--" + gcnarch;
    std::string inputs = "-inputs=/dev/null," + LinkedObjectFile;
    std::string bundledFileStr = "--outputs=" + BundledObjectFile;
    offloadBundleArgList.push_back(clangOffloadBundle.c_str());
    offloadBundleArgList.push_back("-type=o");
    offloadBundleArgList.push_back(inputs.c_str());
    offloadBundleArgList.push_back(targets.c_str());
    offloadBundleArgList.push_back(bundledFileStr.c_str());
    offloadBundleArgList.push_back(nullptr);

    auto cobsra = toStringRefArray(offloadBundleArgList.data());
    sys::ExecuteAndWait(clangOffloadBundle, cobsra);

    ErrorOr<std::unique_ptr<MemoryBuffer>> BundledBinBuf =
        MemoryBuffer::getFile(BundledObjectFile);

    std::string hsaco = BundledBinBuf.get()->getBuffer().str();

    hipModule_t module;
    checkHIP(hipModuleLoadData_p(&module, (const void*)hsaco.c_str()));
    hipFunction_t function;
    checkHIP(hipModuleGetFunction_p(&function, module, "kitsune_kernel"));
    hipStream_t stream;
    hipStreamCreate_p(&stream);
    hipModuleLaunchKernel_p(function, 1, 1, 1, n, 1, 1, 0, stream, args, NULL);

    return (void*) stream;
  }

  void waitHIPKernel(void* wait) {
    hipStreamSynchronize_p((hipStream_t)wait);
  }
}
