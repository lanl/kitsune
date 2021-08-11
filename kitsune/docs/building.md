
This document focuses on the steps required to build the Kitsune+Tapir toolchain. Additional information can be found [here](README.md) for a high-level overview of the Kitsune+Tapir toolchain and how if differs from the base LLVM project's code base. In addition, this [page](USING.md) provides a quick high-level summary of using the Kitsune+Tapir toolchain.

### Building Kitsune+Tapir

Given that Kitsune and Tapir all build upon the LLVM code base, they continue to use the overall LLVM configuration and build approach based on [CMake](https://cmake.org).  There are new options and software package dependencies that are not part of the standard LLVM distribution.  The availability of certain external packages will impact the feature set of the toolchain these details are further discussed below.  The first place to start the process is understanding the basic configuration and build system. 

Those new to building LLVM should take a look at the LLVM
[Getting Started](https://releases.llvm.org/10.0.0/docs/GettingStarted.html) documentation. The remainder of this document assumes you are familiar with the topics discussed in that document.  

**Ninja**: We also encourage use of the [Ninja](https://ninja-build.org) build sytem as it tends to handle parallelism a bit better and thus can result in faster builds.  It also has some additional flexibility in managing the workload between compilation and linking that is very beneficial when building LLVM:

  * The CMake variables ``LLVM_PARALLEL_COMPILE_JOBS`` and ``LLVM_PARALLEL_LINK_JOBS`` can be used to control the number of parallel threads used for compilation and linking.  In general, LLVM's link stages can be extremely memory intensive and setting the number of link job to roughly half the number of compile jobs is a good place to start (it is not uncommon to experience swapping if you are too aggressive in setting the number of link jobs).  If you have a system with significant resources (e.g., hundreds of cores and hundreds of gigabytes of RAM) you can push the limits but smaller systems will struggle during full builds of the toolchain without tweaking these two variables.  A nice balance here can reduce build times to under 10 minutes.  You will have to experiment with the system you are using to find the sweet spot. 

**CMake Cache Files**: We provide two different CMake cache files that can be used to bootstrap the details of configuring the Kitsune and Tapir build.  These files contain comments that describe aspects of the configuration choices and we encourage you to look through them, use them, and tailor them to your needs. 

  * [kitsune-dev.cmake](../cmake/caches/kitsune-dev.cmake) provides a set of options for the basics of building the toolchain.  It is geared towards an in-tree development use case (i.e., skipping the use of an ``install`` target).

  * [kitsune-install.cmake](../cmake/caches/kitsune-install.cmake) provides a production installation set of options. *TODO: Need to create this file.*

The standard CMake command line syntax is used to leverage these cache files.  Note that these files are for the full LLVM build:  
  ```bash
    $ cd $LLVM_SOURCE_DIR    # chage directory to the toplevel directory from git clone of repo.
    $ mkdir build; cd build 
    $ cmake -G Ninja -C ../kitsune/cmake/caches/kitsune-dev.cmake ... ../llvm 
  ``` 
  Although not necessary for most of the LLVM use cases, we encourage using newer versions of CMake.  Specifically versions 3.19 onwards are suggested as the best place to start. *Note* that you may experience warnings using newer releases as older versions are in the process of being phased out and LLVM is using some deprecated features).

**NVIDIA Support**: We strongly encourage using CUDA 11.x with the Kitsune+Tapir toolchain. This will soon become a mandatory requirement as the feature set of CUDA code generation nears completion as it provides some tighter connections with the LLVM components that the toolchain leverages.

  * Also note that there are some very, very old NVIDIA architecture specifications buried inside the LLVM distribution.  This is especially true if you enable the OpenMP runtime as part of the build.  The CMake cache files listed above address this as it can cause the build to fail when combined with the latest CUDA distributions.  In a nutshell, old architecture/capabilities (e.g., *sm_35*) need to be updated to more modern hardware (e.g., *sm_70*). 

**AMD GPU Support**: The support for AMD's latest GPU architectures is under active development.  More details will be provided here as the code matures and lands into a versioned release of the Kitsune+Tapir toolchain. 

**Intel GPU Support**: The support for Intel's GPUs is currently in the queue behind our work on NVIDIA and AMD hardware.

**Module Support**: In favor of internalizing some of the runtime components needed for the toolchain external module support has been disabled in the latest release.  For basic development (i.e., independent of the runtime ABI targets) the basic configure and build settings will include builds of both the Cheetah (OpenCilk) runtime and Kokkos.  Thus is is no longer necessary to have these packages installed prior to configuring and building.  Note, portions of this change are requirements as most of the runtime interfaces will now leverage the generation of version-centric bitcode files that are built directly using the "*in-tree*" toolchain.  As our runtime layers adjust to this design change and a restructuring for post-toolchain builds we will likely reintroduce some aspects of the past module use cases to simplify configuration choices.

