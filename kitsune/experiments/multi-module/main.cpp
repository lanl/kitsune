
// Copyright(c) 2020 Triad National Security, LLC
// All rights reserved.
//
// This file is part of the kitsune / llvm project.  It is released under
// the LLVM license.
//
// Simple example of an element-wise vector sum.
// To enable kitsune+tapir compilation add the flags to a standard
// clang compilation:
//
//    * -ftapir=rt-target : the runtime ABI to target.
//
#include <cstdio>
#include <stdlib.h>
#include <string>
#include <kitsune.h>
#include "kitsune/kitrt/llvm-gpu.h"
#include "kitsune/kitrt/kitrt-cuda.h"

using namespace std;

const size_t VEC_SIZE = 1024 * 1024 * 256;

extern void fill(float *, size_t);
extern void vecadd(const float *A, const float *B, float *C, size_t N);


int main (int argc, char* argv[]) {
  fprintf(stderr, "running multi compiliation unit test...\n");
  size_t size = VEC_SIZE;

  if (argc > 1) {
    size = atol(argv[1]);
  }
  fprintf(stderr, "\tproblem size = %ld\n", size);

  float *A = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *B = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);
  float *C = (float *)__kitrt_cuMemAllocManaged(sizeof(float) * size);

  fill(A, size);
  fill(B, size);
  vecadd(A, B, C, size); 

  fprintf(stderr, "\tchecking results....  ");
  size_t error_count = 0;
  for(size_t i = 0; i < size; i++) {
    float sum = A[i] + B[i];
    if (C[i] != sum)
      error_count++;
  }

  if (error_count > 0)
    fprintf(stderr, "incorrect result, %ld errors found!\n", error_count);
  else 
    fprintf(stderr, "ok, no errors.\n");

  return 0;
}
