
## TODO items for version 13.x:

* Add clang-level flags for setting GPU architecture target details.  For example, ``-ftapir-nvarch=sm_80``.  The current path requires use of the ``-mllvm`` command line option from clang. 

* By default, "borrow" the optimization setting from clang to set the various flags for the ABI transforms; however, would also be nice to add an optional flag for a different level of optimization for ABI tranforms. 

* Need to look at sharing some code bewteen the CUDA and GPU ABIs -- there are some similarities and they should probably at least inherient off a shared base class.

* GPU ABI: Need to generate unique kernel names (see item above).

* The GPU runtime needs to better manage, reuse, and cleanup CUDA resources.  As it stands, a loop of kernel launches will actually create all new CUDA resources (modules, kernels, streams, etc.).

* Add support for systems with multiple GPUs to the runtime.

* Update support for both *just-in-time* and *ahead-of-time* compilation for HIP.

