# Limitations

This describes some of the limitations of Kitsune both in terms of the language
constructs that it supports and the supported backends. Some of the 
limitations may only apply to certain backends. Any limitations that are 
overcome must be removed from here.

## `forall` loop

The constraints described here should also apply when using Kokkos' 
`parallel_for`. 

### Global Variables

Global variables may be used in a `forall` loop. Any value that is not a local
variable declared in the function/method containing the `forall` loop or a
parameter to that function/method is considered a global variable. 

However, such globals are subject to certain constraints.

    - The globals must be compile-time constants and must be declared as such 
      in the source code. In C/C++, this means that they _must_ be declared 
      **`const`**. _[In the future, this restriction may be relaxed to allow 
      for the use of global variables that are only read (but not modified) 
      within the `forall` loop even if they are not declared `const` in the 
      source code]._

    - Even if the global is a constant, it must be a self-contained type. A 
      self-contained type is one that does not contain any pointers. The
      following are considered self-contained types:
      
          - All scalar types (`int`, `float`, `char` etc.)
            - Structs each of whose elements is a self-contained type.
          - Arrays of self-contained types where the array length is known at 
            compile-time.
            
In general, most STL containers in C++ are _not_ self-contained types. Neither
are C++ strings. The std::cout and std::cin objects are global variables and 
thus may not be used in a `forall` loop.

### Constraints

Currently, any functions called from within a `forall` loop must be defined in
the same compilation unit (source file) as the loop. If this restriction is 
relaxed, it will almost certainly then require that the callee also have been
compiled using kitsune. 

### Virtual methods

Virtual method calls from within a `forall` loop are not currently allowed. 

_[This suddenly occurred to me but I haven't thought about whether this is an
actual limitation or not. I just added it here because I thought that it might 
be - Tarun (08 June 2022)]_
