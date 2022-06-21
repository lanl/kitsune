//
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
// 
// 
// Example of operations over an array of complex numbers. 
// 
// To enable kitsune+tapir compilation add the flags to a standard 
// clang compilation: 
//
//    * -ftapir=rt-target : the runtime ABI to target. 
// 
#include <cstdio>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <kitsune.h>
#include "add.h"

using namespace std; 

const size_t VEC_SIZE = 1024; 

int main (int argc, char* argv[]) {
  vector<int> A(VEC_SIZE);
  vector<int> B(VEC_SIZE);
  vector<int> C(VEC_SIZE);

  for(auto i : A) {
    A[i] = rand();
    B[i] = rand();    
  }

  forall(int i=0; i<C.size(); i++){
    C[i] = add(A[i],B[i]);
  }

  size_t ti=0; 
  for(; ti < VEC_SIZE; ++ti) {
    float sum = A[ti] + B[ti];
    if (C[ti] != sum) {
      printf("failure");
      return 1; 
    }
  }
 
  fprintf(stdout, "Result = %s (%ld, %ld)\n",
	  (ti == VEC_SIZE) ? "PASS" : "FAIL",
	  ti, VEC_SIZE);

  return 0;
}


