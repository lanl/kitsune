## Using Kitsune+Tapir 

As the name suggestions there are two components to the Kitsune+Tapir toolchain.  

First, Kitsune provides a set of extensions to Clang that support the front-end components of compiling parallel constructs.  In this case, there are two primary paths to use these language features: 

  * Use of the ``forall`` keyword.  At the level of the syntactic rules a ``forall`` keyword are identical to C/C++ ``for`` loops.  However, the parallel semantics of the loop introduce new restrictions that developers should be aware of:

     * The loop body will execute in parallel where each iteration through the loop should be independent of all others.  Developers should also make no assumptions about any assumed ordering of the iterations in the parallel form of execution.  In a nutshell, these restrictions are no different any many other parallel loop constructs.

  * The parallelization of *some* forms of the [Kokkos]() C++ ``parallel_for`` constructs. This support is enabled by using the ``-fkokkos`` command line argument.  The same rules for parallel loop execution follow both Kokkos and the ``forall`` semantics.  Note that only the lambda form of ``parallel_for`` is recognized by Kitsune.  There are several examples of using Kokkos under the ``kitsune`` directory at the top-level of the source code repository (``kitsune/examples/kokkos``). 

     * Some limited forms of ``Kokkos::parallel_for`` using ``MDRange`` are also supported.  The examples provide a set of currently supported constructs. 

     * ``Kokkos::parallel_reduce`` support is currently under development. **(TODO: This needs to be updated when reductions are completed.)**

*(TODO: Document other language-level constructs -- e.g., ``spawn``, ``sync``.)*

The second component of the compilation stage is Tapir that is implemented as a series of extensions to LLVM.  Tapir is responsible for taking the parallel representations of code provided by Kitsune and (1) optimizing them and (2) taking the optimized code and transforming it into a parallel for for a specific architecture and corresponding runtime ABI target.  The architecture target for the code is primarily defined by the runtime target.  For example: 

  * `-ftapir=serial`: will transform the parallel intermediate form used by Tapir into a serial CPU code. 
  * `-ftapir=opencilk`: will transform the parallel intermediate form used by Tapir into a CPU executable that leverages the OpenCilk runtime system for parallelism. 
  * `-ftapir=openmp`: will transform the parallel intermediate form used by Tapir into a CPU executable that leverages the OpenMP runtime system (even if the input source program is not using OpenMP).
  * `-ftapir=cudatk`: will transform the parallel intermediate form used by Tapir into a runtime target for supporting CUDA and NVIDIA's GPU architectures. `cudatk` is shorthand for a CUDA Toolkit runtime that is part of the Kitsune code base and simplifies some of the details of code generation for CUDA. *(TODO: This transform is still under development and should not be considered robust.)*
  * `-ftapir=hip`: will transform the parallel intermediate form used by Tapir into a runtime target for supporting HIP and AMD's GPU architectures.  Like the CUDA target above, there is a HIP-specific runtime library that is packaged with Kitsune that simplifies some aspects of code generation for the AMD software stack and GPU hardware. 
  * `-ftapir=realm`: will transform the parallel intermediate form used by Tapir into a runtime target that supports the Realm runtime system that is used the low-level runtime system used by the Legion Programming System. 
  * `-ftapir=qthreads`: will transform the parallel intermediate form used by Tapir into a runtime target that supports CPU execution using the Qthreads runtime system. 

  *(TODO: Flush out the details for other, lower priority, targets -- e.g. OpenCL.)* 

Based on the selected runtime and architecture target, the result executable each have their own unique aspects for supporting parallel execution parameters.  Many are controlled by enviornment variables and are dependent upon the runtime system.  A quick overview of some of these parameters are quickly discussed below. 

**OpenCilk Runtime Target**: The OpenCilk runtime target supports one primary environment variable that controls the number of worker threads that will be used to execute supported language constructs (e.g., `forall`).  This environment variable is `CILK_NWORKERS`.  In addition, the compilation step requires a bitcode file for the runtime interface that is currently searched for via the LD_LIBRARY_PATH environment variable.  The bitcode file is installed into ``INSTALL_PREFIX/lib/clang/VERSION/lib/TARGET_TRIPLE`` (see example below) and is also within the LLVM binary/build directory if you are working with an *in-tree* build.

```bash
$ export LD_LIBRARY_PATH=/usr/local/kitsune/lib/clang/10.0.1/lib/x86_64-unknown-linux-gnu:$LD_LIBRARY_PATH
$ clang++ -ftapir=opencilk ... my_program.cpp 
$ export CILK_NWORKERS=16   # use 16 worker threads during execution. 
$ a.out 
``` 

**QThreads Runtime Target**: The Qthreads runtime has [several settings](https://cs.sandia.gov/qthreads/man/qthread_init.html#toc3) via the environment that can impact behavior and 
performance.  At a minimum setting `QTHREAD_NUM_SHEPHERDS` will allow you to control the number of threads assigned to the execution of an executable. 
 ```bash 
 $ clang++ -ftapir=qthreads ... file.cpp 
 $ export QTHREAD_NUMBER_SHEPHERDS=16 # use 16 threads during execution. 
 $ a.out
 ``` 

**Realm Runtime Target**: When running a Realm target you can provide a full command 
line via the `REALM_DEFAULT_ARGS` environment variable. More details on the various 
command line arguments supported by Realm can be found [here](https://legion.stanford.edu/starting/); look for the *Command-Line Flags* section. 

```bash 
$ clang++ -ftapir=realm ... file.cpp 
$ export REALM_DEFAULT_ARGS="-ll:cpu 1 -ll:force_kthreads -level task=2,taskreg=2"
$ a.out 
``` 
**CUDA Toolkit Target**: Still under development and testing... 

**AMD HIP Target**: Still under development and testing... 

*(TODO: Flush out the details for other, lower priority, targets -- e.g. OpenCL.)*

##Compile Options via Configure Files

Unlike previous versions of the toolchain, it is now possible to provide a set of customized configuration files for controlling the various command line arguments used by the compiler for each of the different transformation/ABI targets.  This path avoids the need to hard-code flags and details into the ``clang`` executable.  

There are three search locations the compiler will use to try and locate these files, with the first file found taking precedence over others.  These locations are directories and in priority order are:

  * A user-specific location that is typically stored within their home directory (e.g., ``~/.kitsune-tapir``). 
  * A kitsune-specific location that is typically rolled into the installation as a set of defaults based on configuration and build settings.  These files are normally placed within the LLVM installation under the directory ``share/kitsune/``.
  * Finally, a toolchain-wide location that is specific to clang. 

Each of these locations may be specifically set via CMake when building the toolchain from source. The named variables are:

  * ``CLANG_CONFIG_FILE_USER_DIR``
  * ``CLANG_CONFIG_FILE_KITSUNE_DIR``
  * ``CLANG_CONFIG_FILE_SYSTEM_DIR``

A configuration file using a text format, with options provided exactly as they would be on the command line.  The only other legal syntax within the file are single-line comments that start with a ``#``.  For example: 

```
# Example kitsune+tapir config file. 
-I/projects/kitsune/include -v 
# 
-L/projects/kitsune/lib -lmylibrary 
``` 

Each special mode and runtime transformation/ABI target has its own named configuration file that is specifically searched for at compilation time:

  * ``kokkos.cfg``: Kokkos specific flags to use when ``-fkokkos`` is enabled. 
  * ``opencilk.cfg``: OpenCilk/Cheetah ABI target specific flags. 
  * ``openmp.cfg``: OpenMP runtime ABI target specific flags. 
  * ``qthreads.cfg``: Qthreads runtime ABI target specific flags. 
  * ``realm.cfg``: Realm runtime ABI target specific flags. 
  * ``cudatk.cfg``: Cuda Toolkit runtime ABI target specific flags. 
  * ``hip.cfg``: HIP runtime ABI target specific flags. 
  * ``opencl.cfg``: OpenCL runtime ABI target specific flags. 

These files can reduce complexity for end users by providing configuration- and build-specific flags.  This can be important when version-specific bitcode files and other details are used.  In addition, these files can provide developers additional flexibility for debugging, testing, and experimenting.  Obviously, all these features can also be hardcoded onto the command line for a more traditional use case.  In addition, to override any of the Kitsune or system configuration files you can place an empty config file within the user directory (no kitsune or system configuration files will be read in this case). 

## Reductions 
We provide two approaches to reductions. The first (still very much a work in
progress and likely to break) is implicit reductions. This allows you to write
basic reductions in the way you would for sequential code, and have them be
optimized for parallelism, e.g. 

```
forall(auto x : xs) {
  acc += x;
}
```

should generate efficient parallel reduction code.

Second, we provide a c++ interface for parallel reduction via user-defined
reduction operators. Formally, we require a unital magma, which is just a
reduction operator and a unit value, e.g. 0 for sums and 1 for products.

This allows for the following style of reductions: 

```
#include<reductions.h>
...
  double sum = reduce<double>(Sum<double>(), big);
```



